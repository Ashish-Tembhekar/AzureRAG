import json
import threading
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RequestMetrics:
    request_id: str
    endpoint: str
    started_at: str
    finished_at: Optional[str] = None
    status: str = "in_progress"
    session_id: Optional[str] = None
    index_name: Optional[str] = None
    user_query: Optional[str] = None
    error: Optional[str] = None
    llm: Dict[str, Any] = field(
        default_factory=lambda: {
            "model": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "temperature": None,
            "max_tokens": None,
        }
    )
    vector_store: Dict[str, Any] = field(
        default_factory=lambda: {
            "query_text": None,
            "top_k": 0,
            "used_vector_search": False,
            "used_semantic_ranker": False,
            "semantic_query_count": 0,
            "results_returned": 0,
            "search_text": None,
            "vector_dimensions": 0,
            "result_chunks": [],
        }
    )
    storage: Dict[str, Any] = field(
        default_factory=lambda: {
            "reads": 0,
            "writes": 0,
            "read_operations": [],
            "write_operations": [],
        }
    )
    speech: Dict[str, Any] = field(
        default_factory=lambda: {
            "stt": {
                "input_audio_seconds": 0.0,
                "transcribed_characters": 0,
                "audio_file_path": None,
            },
            "tts": {
                "input_characters": 0,
                "output_audio_seconds": 0.0,
                "audio_bytes": 0,
            },
        }
    )
    response: Dict[str, Any] = field(
        default_factory=lambda: {"answer_characters": 0, "contexts_returned": 0}
    )


@dataclass
class DocumentUploadMetrics:
    upload_id: str
    endpoint: str
    started_at: str
    finished_at: Optional[str] = None
    status: str = "in_progress"
    error: Optional[str] = None
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    file_size_bytes: int = 0
    index_name: Optional[str] = None
    chunking: Dict[str, Any] = field(
        default_factory=lambda: {
            "chunk_size": 0,
            "chunk_overlap": 0,
            "sections_extracted": 0,
            "chunks_created": 0,
            "chunk_ids": [],
            "total_chunk_characters": 0,
        }
    )
    document_intelligence: Dict[str, Any] = field(
        default_factory=lambda: {
            "service": "azure_document_intelligence",
            "model": "prebuilt-layout",
            "pages_processed": 0,
            "page_batches": [],
            "cost_usd": 0.0,
        }
    )
    azure_ai_search: Dict[str, Any] = field(
        default_factory=lambda: {
            "service": "azure_ai_search",
            "index_name": None,
            "index_created": False,
            "indexing_mode": "direct_upload_documents",
            "documents_uploaded": 0,
            "upload_batches": 0,
            "vector_dimensions": 0,
            "estimated_storage_mb": 0.0,
        }
    )
    costs: Dict[str, Any] = field(
        default_factory=lambda: {
            "document_intelligence_cost_usd": 0.0,
            "estimated_total_upload_cost_usd": 0.0,
            "session_cost_summary": {},
        }
    )


class RequestMetricsRecorder:
    def __init__(
        self,
        endpoint: str,
        session_id: Optional[str] = None,
        index_name: Optional[str] = None,
        user_query: Optional[str] = None,
        output_dir: str = "request_metrics",
    ) -> None:
        self._lock = threading.Lock()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = RequestMetrics(
            request_id=uuid4().hex,
            endpoint=endpoint,
            started_at=_utc_now_iso(),
            session_id=session_id,
            index_name=index_name,
            user_query=user_query,
        )

    @property
    def request_id(self) -> str:
        return self.data.request_id

    def update_request(
        self,
        *,
        session_id: Optional[str] = None,
        index_name: Optional[str] = None,
        user_query: Optional[str] = None,
    ) -> None:
        with self._lock:
            if session_id is not None:
                self.data.session_id = session_id
            if index_name is not None:
                self.data.index_name = index_name
            if user_query is not None:
                self.data.user_query = user_query

    def record_llm(
        self,
        *,
        model: Optional[str],
        input_tokens: int,
        output_tokens: int,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> None:
        with self._lock:
            self.data.llm.update(
                {
                    "model": model,
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )

    def record_vector_query(
        self,
        *,
        query_text: str,
        top_k: int,
        used_vector_search: bool,
        used_semantic_ranker: bool,
        search_text: str,
        vector_dimensions: int,
        results: List[Dict[str, Any]],
    ) -> None:
        with self._lock:
            self.data.vector_store.update(
                {
                    "query_text": query_text,
                    "top_k": int(top_k),
                    "used_vector_search": bool(used_vector_search),
                    "used_semantic_ranker": bool(used_semantic_ranker),
                    "semantic_query_count": 1 if used_semantic_ranker else 0,
                    "results_returned": len(results),
                    "search_text": search_text,
                    "vector_dimensions": int(vector_dimensions),
                    "result_chunks": results,
                }
            )

    def record_storage_read(
        self,
        *,
        store: str,
        operation: str,
        item_count: int = 0,
        request_charge: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "store": store,
            "operation": operation,
            "item_count": int(item_count),
            "request_charge": request_charge,
            "details": details or {},
        }
        with self._lock:
            self.data.storage["reads"] += 1
            self.data.storage["read_operations"].append(payload)

    def record_storage_write(
        self,
        *,
        store: str,
        operation: str,
        item_count: int = 1,
        request_charge: Optional[float] = None,
        bytes_written: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "store": store,
            "operation": operation,
            "item_count": int(item_count),
            "request_charge": request_charge,
            "bytes_written": bytes_written,
            "details": details or {},
        }
        with self._lock:
            self.data.storage["writes"] += 1
            self.data.storage["write_operations"].append(payload)

    def record_stt(
        self,
        *,
        audio_file_path: str,
        input_audio_seconds: float,
        transcript: str,
    ) -> None:
        with self._lock:
            self.data.speech["stt"].update(
                {
                    "audio_file_path": audio_file_path,
                    "input_audio_seconds": round(float(input_audio_seconds), 3),
                    "transcribed_characters": len(transcript or ""),
                }
            )

    def record_tts(self, *, text: str, audio_bytes: bytes) -> None:
        with self._lock:
            self.data.speech["tts"].update(
                {
                    "input_characters": len(text or ""),
                    "output_audio_seconds": round(
                        self._estimate_audio_seconds(audio_bytes), 3
                    ),
                    "audio_bytes": len(audio_bytes or b""),
                }
            )

    def record_response(self, *, answer_characters: int, contexts_returned: int) -> None:
        with self._lock:
            self.data.response.update(
                {
                    "answer_characters": int(answer_characters),
                    "contexts_returned": int(contexts_returned),
                }
            )

    def finalize(self, *, status: str, error: Optional[str] = None) -> Path:
        with self._lock:
            self.data.status = status
            self.data.error = error
            self.data.finished_at = _utc_now_iso()
            payload = asdict(self.data)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.output_dir / f"{timestamp}_{self.request_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _estimate_audio_seconds(audio_bytes: bytes) -> float:
        if not audio_bytes:
            return 0.0
        try:
            with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
                frame_rate = wav_file.getframerate() or 0
                frame_count = wav_file.getnframes()
                if frame_rate <= 0:
                    return 0.0
                return frame_count / float(frame_rate)
        except Exception:
            return 0.0


class DocumentUploadMetricsRecorder:
    def __init__(
        self,
        endpoint: str,
        file_name: Optional[str] = None,
        index_name: Optional[str] = None,
        output_dir: str = "upload_metrics",
    ) -> None:
        self._lock = threading.Lock()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = DocumentUploadMetrics(
            upload_id=uuid4().hex,
            endpoint=endpoint,
            started_at=_utc_now_iso(),
            file_name=file_name,
            index_name=index_name,
        )

    def update_file(
        self,
        *,
        file_name: Optional[str] = None,
        file_path: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        index_name: Optional[str] = None,
    ) -> None:
        with self._lock:
            if file_name is not None:
                self.data.file_name = file_name
            if file_path is not None:
                self.data.file_path = file_path
            if file_size_bytes is not None:
                self.data.file_size_bytes = int(file_size_bytes)
            if index_name is not None:
                self.data.index_name = index_name
                self.data.azure_ai_search["index_name"] = index_name

    def record_document_intelligence(
        self, *, pages_processed: int, page_batches: List[str], cost_usd: float
    ) -> None:
        with self._lock:
            self.data.document_intelligence.update(
                {
                    "pages_processed": int(pages_processed),
                    "page_batches": list(page_batches),
                    "cost_usd": round(float(cost_usd), 6),
                }
            )
            self.data.costs["document_intelligence_cost_usd"] = round(
                float(cost_usd), 6
            )

    def record_chunking(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        sections_extracted: int,
        chunks_created: int,
        chunk_ids: List[str],
        total_chunk_characters: int,
    ) -> None:
        with self._lock:
            self.data.chunking.update(
                {
                    "chunk_size": int(chunk_size),
                    "chunk_overlap": int(chunk_overlap),
                    "sections_extracted": int(sections_extracted),
                    "chunks_created": int(chunks_created),
                    "chunk_ids": list(chunk_ids),
                    "total_chunk_characters": int(total_chunk_characters),
                }
            )

    def record_search_upload(
        self,
        *,
        index_name: str,
        index_created: bool,
        documents_uploaded: int,
        vector_dimensions: int,
        estimated_storage_mb: float,
        upload_batches: int = 1,
    ) -> None:
        with self._lock:
            self.data.azure_ai_search.update(
                {
                    "index_name": index_name,
                    "index_created": bool(index_created),
                    "documents_uploaded": int(documents_uploaded),
                    "upload_batches": int(upload_batches),
                    "vector_dimensions": int(vector_dimensions),
                    "estimated_storage_mb": round(float(estimated_storage_mb), 6),
                }
            )

    def record_cost_summary(self, *, cost_summary: Dict[str, Any], upload_cost_usd: float) -> None:
        with self._lock:
            self.data.costs["session_cost_summary"] = dict(cost_summary)
            self.data.costs["estimated_total_upload_cost_usd"] = round(
                float(upload_cost_usd), 6
            )

    def finalize(self, *, status: str, error: Optional[str] = None) -> Path:
        with self._lock:
            self.data.status = status
            self.data.error = error
            self.data.finished_at = _utc_now_iso()
            payload = asdict(self.data)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.output_dir / f"{timestamp}_{self.data.upload_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
