import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from azure.core.exceptions import ResourceNotFoundError
from PyPDF2 import PdfReader


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
    input_per_1k_tokens_usd: float = float(os.getenv("AZURE_LLM_INPUT_COST_PER_1K", "0.00015"))
    output_per_1k_tokens_usd: float = float(os.getenv("AZURE_LLM_OUTPUT_COST_PER_1K", "0.00060"))
    semantic_ranker_per_query_usd: float = float(os.getenv("AZURE_SEMANTIC_QUERY_COST", "0.001"))


@dataclass
class UsageTracker:
    total_session_cost_usd: float = 0.0
    semantic_queries_used: int = 0
    storage_used_mb: float = 0.0
    adi_pages_used_this_session: int = 0
    total_adi_pages_month: int = 0
    adi_cost_usd: float = 0.0

    def as_json_summary(self, query_cost_usd: float, limit_status: str) -> Dict[str, Any]:
        return {
            "query_cost_usd": round(query_cost_usd, 6),
            "total_session_cost_usd": round(self.total_session_cost_usd, 6),
            "semantic_queries_used": int(self.semantic_queries_used),
            "storage_used_mb": round(self.storage_used_mb, 6),
            "adi_pages_used_this_session": int(self.adi_pages_used_this_session),
            "total_adi_pages_month": int(self.total_adi_pages_month),
            "adi_cost_usd": round(self.adi_cost_usd, 6),
            "limit_status": limit_status,
        }


class AzureCapacityMonitor:
    _instance: Optional["AzureCapacityMonitor"] = None
    _instance_lock = threading.Lock()

    MAX_ADI_PAGES_PER_MONTH = 500
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
        self._counter_path = Path(".semantic_query_counter.json")
        self._semantic_counter = self._load_semantic_counter()
        self._sync_semantic_usage()
        self._adi_counter_path = Path(".adi_page_counter.json")
        self._adi_counter = self._load_adi_counter()
        self._sync_adi_usage()
        self._initialized = True

    @staticmethod
    def _resolve_tier() -> str:
        tier = os.getenv("AZURE_TIER", "FREE").strip().upper()
        return tier if tier in {"FREE", "BASIC"} else "FREE"

    @property
    def limits(self) -> TierLimits:
        return self.FREE_LIMITS if self.tier == "FREE" else self.BASIC_LIMITS

    def _refresh_tier_locked(self) -> None:
        self.tier = self._resolve_tier()

    @staticmethod
    def _month_key() -> str:
        now = datetime.now(timezone.utc)
        return f"{now.year:04d}-{now.month:02d}"

    def _load_semantic_counter(self) -> Dict[str, Any]:
        if not self._counter_path.exists():
            return {"month": self._month_key(), "count": 0}
        try:
            payload = json.loads(self._counter_path.read_text(encoding="utf-8"))
            if "month" not in payload or "count" not in payload:
                return {"month": self._month_key(), "count": 0}
            return payload
        except Exception:
            return {"month": self._month_key(), "count": 0}

    def _save_semantic_counter(self) -> None:
        self._counter_path.write_text(json.dumps(self._semantic_counter), encoding="utf-8")

    def _sync_semantic_usage(self) -> None:
        self._semantic_counter = self._load_semantic_counter()
        current = self._month_key()
        if self._semantic_counter.get("month") != current:
            self._semantic_counter = {"month": current, "count": 0}
            self._save_semantic_counter()
        self.usage.semantic_queries_used = int(self._semantic_counter.get("count", 0))

    def _load_adi_counter(self) -> Dict[str, Any]:
        if not self._adi_counter_path.exists():
            return {"month": self._month_key(), "count": 0}
        try:
            payload = json.loads(self._adi_counter_path.read_text(encoding="utf-8"))
            if "month" not in payload or "count" not in payload:
                return {"month": self._month_key(), "count": 0}
            return payload
        except Exception:
            return {"month": self._month_key(), "count": 0}

    def _save_adi_counter(self) -> None:
        self._adi_counter_path.write_text(json.dumps(self._adi_counter), encoding="utf-8")

    def _sync_adi_usage(self) -> None:
        self._adi_counter = self._load_adi_counter()
        current = self._month_key()
        if self._adi_counter.get("month") != current:
            self._adi_counter = {"month": current, "count": 0}
            self._save_adi_counter()
        self.usage.total_adi_pages_month = int(self._adi_counter.get("count", 0))

    def refresh_tier(self) -> None:
        with self._lock:
            self._refresh_tier_locked()

    def reset_usage(self) -> None:
        with self._lock:
            self.usage = UsageTracker()
            self._semantic_counter = {"month": self._month_key(), "count": 0}
            self._save_semantic_counter()
            self._adi_counter = {"month": self._month_key(), "count": 0}
            self._save_adi_counter()

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
        if self.MAX_ADI_PAGES_PER_MONTH:
            ratios.append(self.usage.total_adi_pages_month / self.MAX_ADI_PAGES_PER_MONTH)
        if ratios and max(ratios) >= 0.9:
            return "FREE_NEAR_LIMIT"
        return "FREE_OK"

    def _preflight_index_creation_locked(self, index_client: Any, index_name: str) -> None:
        names = list(index_client.list_index_names())
        if index_name not in names and len(names) >= (self.limits.max_indexes or 0):
            self._raise_quota(
                f"Index limit exceeded: existing={len(names)}, max={self.limits.max_indexes}"
            )

    def preflight_index_creation(self, index_client: Any, index_name: str) -> None:
        with self._lock:
            self._refresh_tier_locked()
            if self.tier != "FREE":
                return
            self._preflight_index_creation_locked(index_client=index_client, index_name=index_name)

    def _get_index_stats(self, index_client: Any, index_name: str) -> tuple[int, float]:
        try:
            stats = index_client.get_index_statistics(index_name)
            if isinstance(stats, dict):
                docs = int(
                    stats.get("document_count", stats.get("documentCount", 0)) or 0
                )
                storage_bytes = float(
                    stats.get("storage_size", stats.get("storageSize", 0.0)) or 0.0
                )
            else:
                docs = int(getattr(stats, "document_count", 0) or 0)
                storage_bytes = float(getattr(stats, "storage_size", 0.0) or 0.0)
            storage_mb = storage_bytes / (1024.0 * 1024.0)
            return docs, storage_mb
        except ResourceNotFoundError:
            return 0, 0.0

    def preflight_ingestion(
        self,
        index_client: Any,
        search_client: Any,
        index_name: str,
        documents_added: int,
        storage_added_mb: float,
        is_new_index: bool,
    ) -> None:
        with self._lock:
            self._refresh_tier_locked()
            if self.tier != "FREE":
                return

            if is_new_index:
                self._preflight_index_creation_locked(index_client=index_client, index_name=index_name)

            _, current_storage_mb = self._get_index_stats(index_client, index_name)
            current_docs = int(search_client.get_document_count())
            projected_docs = current_docs + documents_added
            projected_storage_mb = current_storage_mb + storage_added_mb

            if projected_docs > (self.limits.max_documents_per_index or 0):
                self._raise_quota(
                    "Documents per index limit exceeded: "
                    f"index={index_name}, attempted={projected_docs}, max={self.limits.max_documents_per_index}"
                )
            if projected_storage_mb > (self.limits.max_storage_mb or 0):
                self._raise_quota(
                    "Storage limit exceeded: "
                    f"index={index_name}, attempted={projected_storage_mb:.6f} MB, max={self.limits.max_storage_mb} MB"
                )

    def preflight_semantic_query(self) -> None:
        with self._lock:
            self._refresh_tier_locked()
            self._sync_semantic_usage()
            projected = self.usage.semantic_queries_used + 1
            if self.tier == "FREE" and projected > (self.limits.max_semantic_ranker_queries or 0):
                self._raise_quota(
                    "Semantic ranker query limit exceeded: "
                    f"attempted={projected}, max={self.limits.max_semantic_ranker_queries}"
                )
            self._semantic_counter["count"] = projected
            self._save_semantic_counter()
            self.usage.semantic_queries_used = projected

    @staticmethod
    def _count_document_pages(file_path: str) -> int:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document path not found: {file_path}")
        if path.suffix.lower() == ".pdf":
            with path.open("rb") as handle:
                try:
                    reader = PdfReader(handle)
                    if reader.is_encrypted:
                        try:
                            reader.decrypt("")
                        except Exception as exc:
                            raise ValueError(
                                "Encrypted PDF detected. Please provide an unencrypted PDF."
                            ) from exc
                    return len(reader.pages)
                except Exception as exc:
                    raise ValueError(
                        "Unable to read PDF pages for quota validation. "
                        "If this PDF is encrypted, install pycryptodome and provide an unencrypted file."
                    ) from exc
        return 1

    def verify_adi_page_quota(self, file_path: str) -> int:
        with self._lock:
            self._refresh_tier_locked()
            self._sync_adi_usage()
            pages = self._count_document_pages(file_path)
            if pages <= 0:
                raise ValueError("Document page count could not be determined.")
            projected = self.usage.total_adi_pages_month + pages
            if self.tier == "FREE" and projected > self.MAX_ADI_PAGES_PER_MONTH:
                self._raise_quota(
                    "Document Intelligence page limit exceeded: "
                    f"attempted={projected}, max={self.MAX_ADI_PAGES_PER_MONTH}"
                )
            return pages

    def record_ingestion(self, storage_added_mb: float) -> str:
        with self._lock:
            self.usage.storage_used_mb += storage_added_mb
            return self._limit_status()

    def register_adi_pages(self, pages: int, adi_cost_usd: float) -> str:
        with self._lock:
            self._refresh_tier_locked()
            self._sync_adi_usage()
            pages = max(int(pages), 0)
            if pages:
                projected = self.usage.total_adi_pages_month + pages
                self._adi_counter["count"] = projected
                self._save_adi_counter()
                self.usage.total_adi_pages_month = projected
                self.usage.adi_pages_used_this_session += pages
            self.usage.adi_cost_usd += adi_cost_usd
            self.usage.total_session_cost_usd += adi_cost_usd
            return self._limit_status()

    def register_query(self, query_cost_usd: float) -> str:
        with self._lock:
            self.usage.total_session_cost_usd += query_cost_usd
            self._sync_semantic_usage()
            self._sync_adi_usage()
            return self._limit_status()


class RAGCostEvaluator:
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
        embedding_bytes_added: int,
        adi_pages_used: int = 0,
    ) -> Dict[str, Any]:
        storage_added_mb = self._bytes_to_mb(embedding_bytes_added)
        limit_status = self.capacity_monitor.record_ingestion(storage_added_mb=storage_added_mb)
        adi_cost_usd = 0.0
        if adi_pages_used:
            self.capacity_monitor.refresh_tier()
            if self.capacity_monitor.tier == "BASIC":
                adi_cost_usd = (adi_pages_used / 1000.0) * 10.0
            limit_status = self.capacity_monitor.register_adi_pages(
                pages=0,
                adi_cost_usd=adi_cost_usd,
            )
        return self.capacity_monitor.usage.as_json_summary(query_cost_usd=0.0, limit_status=limit_status)

    def evaluate_query(
        self,
        input_tokens: int,
        output_tokens: int,
        use_semantic_ranker: bool = False,
    ) -> Dict[str, Any]:
        llm_cost = (input_tokens / 1000.0) * self.pricing.input_per_1k_tokens_usd
        llm_cost += (output_tokens / 1000.0) * self.pricing.output_per_1k_tokens_usd
        semantic_cost = self.pricing.semantic_ranker_per_query_usd if use_semantic_ranker else 0.0
        query_cost = llm_cost + semantic_cost
        limit_status = self.capacity_monitor.register_query(query_cost_usd=query_cost)
        return self.capacity_monitor.usage.as_json_summary(
            query_cost_usd=query_cost,
            limit_status=limit_status,
        )


_evaluator_singleton: Optional[RAGCostEvaluator] = None
_evaluator_lock = threading.Lock()


def get_rag_cost_evaluator() -> RAGCostEvaluator:
    global _evaluator_singleton
    with _evaluator_lock:
        if _evaluator_singleton is None:
            _evaluator_singleton = RAGCostEvaluator()
    return _evaluator_singleton
