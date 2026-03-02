import hashlib
import math
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import QueryType, VectorizedQuery
from dotenv import load_dotenv

from .azure_simulator import AzureCapacityMonitor

load_dotenv()


@dataclass
class ChunkRecord:
    chunk_id: str
    index_name: str
    source_name: str
    text: str
    metadata: str


class AzureSearchDocumentStore:
    """Real Azure AI Search-backed document store."""

    def __init__(self) -> None:
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "").strip()
        admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY", "").strip()
        if not endpoint or not admin_key:
            raise RuntimeError(
                "AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_ADMIN_KEY must be set in .env."
            )

        self._lock = threading.Lock()
        self._endpoint = endpoint
        self._credential = AzureKeyCredential(admin_key)
        self._index_client = SearchIndexClient(endpoint=endpoint, credential=self._credential)
        self._search_clients: Dict[str, SearchClient] = {}
        self._capacity_monitor = AzureCapacityMonitor()

    @staticmethod
    def _is_text_like(chunk: str) -> bool:
        if not chunk:
            return False
        total = len(chunk)
        printable = sum(1 for c in chunk if c.isprintable() or c in "\n\r\t")
        alpha_num = sum(1 for c in chunk if c.isalnum())
        printable_ratio = printable / total
        alpha_num_ratio = alpha_num / total
        return printable_ratio >= 0.85 and alpha_num_ratio >= 0.15

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        clean = " ".join(text.split())
        if not clean:
            return []

        chunks: List[str] = []
        start = 0
        text_len = len(clean)
        step = max(chunk_size - overlap, 1)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            candidate = clean[start:end]
            if AzureSearchDocumentStore._is_text_like(candidate):
                chunks.append(candidate)
            if end >= text_len:
                break
            start += step
        return chunks

    def _search_client(self, index_name: str) -> SearchClient:
        with self._lock:
            client = self._search_clients.get(index_name)
            if client is None:
                client = SearchClient(
                    endpoint=self._endpoint,
                    index_name=index_name,
                    credential=self._credential,
                )
                self._search_clients[index_name] = client
            return client

    @staticmethod
    def _embed_text_deterministic(text: str, dimensions: int) -> List[float]:
        """
        Deterministic local embedding so vector search executes against real Azure Search.
        """
        vec = [0.0] * dimensions
        if not text:
            return vec

        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:4], "little") % dimensions
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[slot] += sign

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _safe_doc_key(raw: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_=-]", "-", raw)
        return sanitized[:1024]

    def ensure_index(self, index_name: str, vector_dimensions: int) -> bool:
        try:
            self._index_client.get_index(index_name)
            return False
        except ResourceNotFoundError:
            self._capacity_monitor.preflight_index_creation(self._index_client, index_name)

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="chunk", type=SearchFieldDataType.String),
            SearchableField(name="source_name", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="metadata", type=SearchFieldDataType.String),
            SimpleField(
                name="chunk_number",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="created_at",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True,
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=vector_dimensions,
                vector_search_profile_name="vector-profile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
            profiles=[
                VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-config",
                )
            ],
        )
        semantic_search = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="semantic-config",
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=None,
                        content_fields=[SemanticField(field_name="chunk")],
                        keywords_fields=[SemanticField(field_name="source_name")],
                    ),
                )
            ]
        )

        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )
        self._index_client.create_index(index)
        return True

    def ingest_text(
        self,
        index_name: str,
        source_name: str,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        vector_dimensions: int = 1536,
    ) -> List[ChunkRecord]:
        index_name = index_name.strip().lower()
        created = self.ensure_index(index_name=index_name, vector_dimensions=vector_dimensions)

        chunks = self.chunk_text(text=text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return []

        docs = []
        now = datetime.now(timezone.utc)
        for idx, chunk in enumerate(chunks, start=1):
            chunk_id = self._safe_doc_key(f"{index_name}-{source_name}-{idx}-{abs(hash(chunk))}")
            docs.append(
                {
                    "id": chunk_id,
                    "chunk": chunk,
                    "source_name": source_name,
                    "metadata": f'{{"index":"{index_name}","source":"{source_name}"}}',
                    "chunk_number": idx,
                    "created_at": now.isoformat(),
                    "content_vector": self._embed_text_deterministic(
                        text=chunk, dimensions=vector_dimensions
                    ),
                }
            )

        estimated_storage_mb = (
            len(docs) * vector_dimensions * 4.0 / (1024.0 * 1024.0)
        )
        client = self._search_client(index_name)
        self._capacity_monitor.preflight_ingestion(
            index_client=self._index_client,
            search_client=client,
            index_name=index_name,
            documents_added=len(docs),
            storage_added_mb=estimated_storage_mb,
            is_new_index=created,
        )

        result = client.upload_documents(documents=docs)
        failed = [r for r in result if not r.succeeded]
        if failed:
            raise RuntimeError(f"upload_documents failed for {len(failed)} chunks.")

        return [
            ChunkRecord(
                chunk_id=d["id"],
                index_name=index_name,
                source_name=source_name,
                text=d["chunk"],
                metadata=d["metadata"],
            )
            for d in docs
        ]

    def search(
        self,
        query: str,
        index_name: Optional[str] = None,
        top_k: int = 4,
        use_semantic_ranker: bool = False,
        use_vector_search: bool = True,
        vector_dimensions: int = 1536,
    ) -> List[ChunkRecord]:
        if not index_name:
            raise ValueError("index_name is required for real Azure Search queries.")

        index_name = index_name.strip().lower()
        client = self._search_client(index_name)
        search_kwargs: Dict[str, Any] = {"top": max(top_k, 1)}

        if use_semantic_ranker:
            self._capacity_monitor.preflight_semantic_query()
            search_kwargs["query_type"] = QueryType.SEMANTIC
            search_kwargs["semantic_configuration_name"] = "semantic-config"
            search_kwargs["query_caption"] = "extractive|highlight-false"
            search_kwargs["query_answer"] = "extractive|count-1"

        if use_vector_search:
            query_vector = self._embed_text_deterministic(query, dimensions=vector_dimensions)
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=max(top_k, 1),
                fields="content_vector",
            )
            search_kwargs["vector_queries"] = [vector_query]

        results = client.search(search_text=query or "*", **search_kwargs)
        payload: List[ChunkRecord] = []
        for item in results:
            chunk_text = item.get("chunk", "")
            if not self._is_text_like(chunk_text):
                continue
            payload.append(
                ChunkRecord(
                    chunk_id=item["id"],
                    index_name=index_name,
                    source_name=item.get("source_name", ""),
                    text=chunk_text,
                    metadata=item.get("metadata", ""),
                )
            )
        return payload

    @staticmethod
    def _escape_odata_string(value: str) -> str:
        return value.replace("'", "''")

    def list_documents(self, index_name: str, top: int = 1000) -> List[Dict[str, Any]]:
        index_name = index_name.strip().lower()
        client = self._search_client(index_name)
        results = client.search(
            search_text="*",
            top=max(top, 1),
            select=["id", "source_name", "created_at", "chunk_number"],
        )
        grouped: Dict[str, Dict[str, Any]] = {}
        for item in results:
            source_name = str(item.get("source_name", "")).strip() or "unknown"
            entry = grouped.setdefault(
                source_name,
                {
                    "source_name": source_name,
                    "index_name": index_name,
                    "chunk_count": 0,
                    "latest_created_at": "",
                },
            )
            entry["chunk_count"] += 1
            created = item.get("created_at")
            created_text = created.isoformat() if hasattr(created, "isoformat") else str(created or "")
            if created_text and created_text > entry["latest_created_at"]:
                entry["latest_created_at"] = created_text

        docs = list(grouped.values())
        docs.sort(key=lambda x: (x["latest_created_at"], x["source_name"]), reverse=True)
        return docs

    def delete_by_sources(self, index_name: str, source_names: List[str]) -> Dict[str, Any]:
        index_name = index_name.strip().lower()
        client = self._search_client(index_name)
        unique_sources = [s.strip() for s in source_names if s and s.strip()]
        total_deleted = 0

        for source_name in unique_sources:
            escaped = self._escape_odata_string(source_name)
            results = client.search(
                search_text="*",
                filter=f"source_name eq '{escaped}'",
                select=["id"],
                top=1000,
            )
            ids = [{"id": item["id"]} for item in results if item.get("id")]
            if not ids:
                continue
            response = client.delete_documents(documents=ids)
            deleted_now = sum(1 for r in response if getattr(r, "succeeded", False))
            total_deleted += deleted_now

        return {"deleted_documents": total_deleted, "sources_requested": len(unique_sources)}

    def stats(self) -> Dict[str, Any]:
        indexes: Dict[str, Any] = {}
        total_docs = 0
        total_storage_mb = 0.0
        for name in self._index_client.list_index_names():
            stat = self._index_client.get_index_statistics(name)
            doc_count = int(self._search_client(name).get_document_count())
            if isinstance(stat, dict):
                storage_bytes = float(
                    stat.get("storage_size", stat.get("storageSize", 0.0)) or 0.0
                )
            else:
                storage_bytes = float(getattr(stat, "storage_size", 0.0) or 0.0)
            storage_mb = storage_bytes / (1024.0 * 1024.0)
            indexes[name] = {
                "documents": doc_count,
                "storage_mb": round(storage_mb, 6),
            }
            total_docs += doc_count
            total_storage_mb += storage_mb
        return {
            "indexes": indexes,
            "total_documents": total_docs,
            "total_storage_mb": round(total_storage_mb, 6),
        }


_store_singleton: Optional[AzureSearchDocumentStore] = None
_store_lock = threading.Lock()


def get_document_store() -> AzureSearchDocumentStore:
    global _store_singleton
    with _store_lock:
        if _store_singleton is None:
            _store_singleton = AzureSearchDocumentStore()
    return _store_singleton
