import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class AzureQuotaExceededError(RuntimeError):
    """Raised when Azure FREE tier capacity constraints are exceeded."""


@dataclass(frozen=True)
class TierLimits:
    max_indexes: Optional[int]
    max_documents_per_index: Optional[int]
    max_storage_mb: Optional[float]
    max_semantic_ranker_queries: Optional[int]


@dataclass
class ModelPricing:
    # Defaults are configurable to avoid hardcoding model-dependent values.
    input_per_1k_tokens_usd: float = float(os.getenv("AZURE_LLM_INPUT_COST_PER_1K", "0.00015"))
    output_per_1k_tokens_usd: float = float(os.getenv("AZURE_LLM_OUTPUT_COST_PER_1K", "0.00060"))
    semantic_ranker_per_query_usd: float = float(os.getenv("AZURE_SEMANTIC_QUERY_COST", "0.001"))


@dataclass
class UsageTracker:
    total_session_cost_usd: float = 0.0
    semantic_queries_used: int = 0
    storage_used_mb: float = 0.0
    indexes: set[str] = field(default_factory=set)
    documents_per_index: dict[str, int] = field(default_factory=dict)

    def as_json_summary(self, query_cost_usd: float, limit_status: str) -> Dict[str, Any]:
        return {
            "query_cost_usd": round(query_cost_usd, 6),
            "total_session_cost_usd": round(self.total_session_cost_usd, 6),
            "semantic_queries_used": self.semantic_queries_used,
            "storage_used_mb": round(self.storage_used_mb, 6),
            "limit_status": limit_status,
        }


class AzureCapacityMonitor:
    """
    Singleton monitor that enforces FREE tier hard limits.
    BASIC tier bypasses hard limits and only accumulates usage/cost telemetry.
    """

    _instance: Optional["AzureCapacityMonitor"] = None
    _instance_lock = threading.Lock()

    FREE_LIMITS = TierLimits(
        max_indexes=3,
        max_documents_per_index=1000,
        max_storage_mb=50.0,
        max_semantic_ranker_queries=1000,
    )
    BASIC_LIMITS = TierLimits(
        max_indexes=None,
        max_documents_per_index=None,
        max_storage_mb=None,
        max_semantic_ranker_queries=None,
    )

    def __new__(cls) -> "AzureCapacityMonitor":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._lock = threading.Lock()
        self.usage = UsageTracker()
        self.tier = self._resolve_tier()
        self._initialized = True

    @staticmethod
    def _resolve_tier() -> str:
        tier = os.getenv("AZURE_TIER", "FREE").strip().upper()
        return tier if tier in {"FREE", "BASIC"} else "FREE"

    @property
    def limits(self) -> TierLimits:
        return self.FREE_LIMITS if self.tier == "FREE" else self.BASIC_LIMITS

    def refresh_tier(self) -> None:
        with self._lock:
            self.tier = self._resolve_tier()

    def reset_usage(self) -> None:
        with self._lock:
            self.usage = UsageTracker()

    def _raise_quota(self, message: str) -> None:
        raise AzureQuotaExceededError(f"[{self.tier}] {message}")

    def _limit_status(self) -> str:
        if self.tier == "BASIC":
            return "BASIC_UNLIMITED"

        limits = self.limits
        ratios = []
        if limits.max_storage_mb:
            ratios.append(self.usage.storage_used_mb / limits.max_storage_mb)
        if limits.max_semantic_ranker_queries:
            ratios.append(self.usage.semantic_queries_used / limits.max_semantic_ranker_queries)
        if limits.max_indexes:
            ratios.append(len(self.usage.indexes) / limits.max_indexes)
        if ratios and max(ratios) >= 0.9:
            return "FREE_NEAR_LIMIT"
        return "FREE_OK"

    def register_ingestion(
        self,
        index_name: str,
        documents_added: int,
        storage_added_mb: float,
    ) -> str:
        with self._lock:
            if self.tier == "FREE":
                limits = self.limits
                if index_name not in self.usage.indexes:
                    projected_indexes = len(self.usage.indexes) + 1
                    if projected_indexes > (limits.max_indexes or 0):
                        self._raise_quota(
                            f"Index limit exceeded: attempted {projected_indexes}, max {limits.max_indexes}"
                        )

                existing_docs = self.usage.documents_per_index.get(index_name, 0)
                projected_docs = existing_docs + documents_added
                if projected_docs > (limits.max_documents_per_index or 0):
                    self._raise_quota(
                        "Documents per index limit exceeded: "
                        f"index={index_name}, attempted {projected_docs}, max {limits.max_documents_per_index}"
                    )

                projected_storage = self.usage.storage_used_mb + storage_added_mb
                if projected_storage > (limits.max_storage_mb or 0):
                    self._raise_quota(
                        f"Storage limit exceeded: attempted {projected_storage:.6f} MB, "
                        f"max {limits.max_storage_mb} MB"
                    )

            self.usage.indexes.add(index_name)
            self.usage.documents_per_index[index_name] = (
                self.usage.documents_per_index.get(index_name, 0) + documents_added
            )
            self.usage.storage_used_mb += storage_added_mb
            return self._limit_status()

    def register_query(self, query_cost_usd: float, used_semantic_ranker: bool) -> str:
        with self._lock:
            if self.tier == "FREE" and used_semantic_ranker:
                limits = self.limits
                projected_semantic = self.usage.semantic_queries_used + 1
                if projected_semantic > (limits.max_semantic_ranker_queries or 0):
                    self._raise_quota(
                        "Semantic ranker query limit exceeded: "
                        f"attempted {projected_semantic}, max {limits.max_semantic_ranker_queries}"
                    )

            if used_semantic_ranker:
                self.usage.semantic_queries_used += 1
            self.usage.total_session_cost_usd += query_cost_usd
            return self._limit_status()


class RAGCostEvaluator:
    """Cost and quota evaluator for ingestion/search operations."""

    def __init__(
        self,
        capacity_monitor: Optional[AzureCapacityMonitor] = None,
        pricing: Optional[ModelPricing] = None,
    ) -> None:
        self.capacity_monitor = capacity_monitor or AzureCapacityMonitor()
        self.pricing = pricing or ModelPricing()

    @staticmethod
    def _bytes_to_mb(num_bytes: int) -> float:
        return float(num_bytes) / (1024.0 * 1024.0)

    def evaluate_ingestion(
        self,
        index_name: str,
        documents_added: int,
        embedding_bytes_added: int,
    ) -> Dict[str, Any]:
        storage_added_mb = self._bytes_to_mb(embedding_bytes_added)
        limit_status = self.capacity_monitor.register_ingestion(
            index_name=index_name,
            documents_added=documents_added,
            storage_added_mb=storage_added_mb,
        )
        return self.capacity_monitor.usage.as_json_summary(
            query_cost_usd=0.0,
            limit_status=limit_status,
        )

    def evaluate_query(
        self,
        input_tokens: int,
        output_tokens: int,
        use_semantic_ranker: bool = False,
    ) -> Dict[str, Any]:
        llm_cost = (input_tokens / 1000.0) * self.pricing.input_per_1k_tokens_usd
        llm_cost += (output_tokens / 1000.0) * self.pricing.output_per_1k_tokens_usd
        agentic_retrieval_cost = (
            self.pricing.semantic_ranker_per_query_usd if use_semantic_ranker else 0.0
        )
        query_cost = llm_cost + agentic_retrieval_cost

        limit_status = self.capacity_monitor.register_query(
            query_cost_usd=query_cost,
            used_semantic_ranker=use_semantic_ranker,
        )
        return self.capacity_monitor.usage.as_json_summary(
            query_cost_usd=query_cost,
            limit_status=limit_status,
        )


_evaluator_singleton: Optional[RAGCostEvaluator] = None
_evaluator_lock = threading.Lock()


def get_rag_cost_evaluator() -> RAGCostEvaluator:
    """Dependency factory usable directly from FastAPI routes."""
    global _evaluator_singleton
    with _evaluator_lock:
        if _evaluator_singleton is None:
            _evaluator_singleton = RAGCostEvaluator()
    return _evaluator_singleton
