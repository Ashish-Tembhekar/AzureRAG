import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChunkRecord:
    chunk_id: str
    index_name: str
    source_name: str
    text: str


@dataclass
class IndexStore:
    chunks: List[ChunkRecord] = field(default_factory=list)


class InMemoryDocumentStore:
    """Simple in-memory document/chunk store for simulator workloads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._indexes: Dict[str, IndexStore] = {}

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
            chunks.append(clean[start:end])
            if end >= text_len:
                break
            start += step
        return chunks

    def ingest_text(
        self,
        index_name: str,
        source_name: str,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> List[ChunkRecord]:
        chunks = self.chunk_text(text=text, chunk_size=chunk_size, overlap=overlap)
        with self._lock:
            index = self._indexes.setdefault(index_name, IndexStore())
            created: List[ChunkRecord] = []
            for chunk in chunks:
                chunk_id = f"{index_name}-{len(index.chunks) + 1}"
                record = ChunkRecord(
                    chunk_id=chunk_id,
                    index_name=index_name,
                    source_name=source_name,
                    text=chunk,
                )
                index.chunks.append(record)
                created.append(record)
            return created

    def search(
        self,
        query: str,
        index_name: Optional[str] = None,
        top_k: int = 4,
    ) -> List[ChunkRecord]:
        with self._lock:
            if index_name:
                indexes = [self._indexes.get(index_name)] if index_name in self._indexes else []
            else:
                indexes = list(self._indexes.values())

            all_chunks: List[ChunkRecord] = []
            for idx in indexes:
                if idx is not None:
                    all_chunks.extend(idx.chunks)

        query_terms = {t for t in query.lower().split() if t}
        scored: List[tuple[float, ChunkRecord]] = []
        for chunk in all_chunks:
            text_lower = chunk.text.lower()
            overlap_score = sum(1 for term in query_terms if term in text_lower)
            density_bonus = (overlap_score / max(len(query_terms), 1)) if query_terms else 0
            starts_bonus = 0.2 if text_lower.startswith(query.lower().strip()) else 0.0
            score = float(overlap_score) + density_bonus + starts_bonus
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:max(top_k, 1)]]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            payload: Dict[str, Any] = {"indexes": {}, "total_chunks": 0}
            total = 0
            for name, index in self._indexes.items():
                count = len(index.chunks)
                payload["indexes"][name] = {"chunks": count}
                total += count
            payload["total_chunks"] = total
            return payload


_store_singleton: Optional[InMemoryDocumentStore] = None
_store_lock = threading.Lock()


def get_document_store() -> InMemoryDocumentStore:
    global _store_singleton
    with _store_lock:
        if _store_singleton is None:
            _store_singleton = InMemoryDocumentStore()
    return _store_singleton
