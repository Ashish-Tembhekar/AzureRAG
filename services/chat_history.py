import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError
from dotenv import load_dotenv

from .azure_simulator import AzureCapacityMonitor
from .request_metrics import RequestMetricsRecorder

load_dotenv()


class CosmosChatHistoryService:
    _instance: Optional["CosmosChatHistoryService"] = None
    _lock = threading.Lock()

    DATABASE_ID = "RagChatDB"
    CONTAINER_ID = "Conversations"

    def __new__(cls) -> "CosmosChatHistoryService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")
        key = os.getenv("AZURE_COSMOS_KEY")
        if not endpoint or not key:
            raise RuntimeError(
                "AZURE_COSMOS_ENDPOINT and AZURE_COSMOS_KEY must be set in .env"
            )
        self._client = CosmosClient(endpoint, credential=key)
        self._database = self._client.get_database_client(self.DATABASE_ID)
        self._container = self._database.get_container_client(self.CONTAINER_ID)
        self._capacity_monitor = AzureCapacityMonitor()
        self._last_request_charge = 0.0
        self._initialized = True

    def _create_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        self._last_request_charge = 0.0
        try:
            response = self._container.create_item(item)
            if hasattr(response, "headers"):
                self._last_request_charge = float(
                    response.headers.get("x-ms-request-charge", 0.0)
                )
            elif hasattr(response, "__dict__"):
                headers = getattr(response, "headers", {})
                if headers and hasattr(headers, "get"):
                    self._last_request_charge = float(
                        headers.get("x-ms-request-charge", 0.0)
                    )
            return response
        except CosmosHttpResponseError as exc:
            if exc.headers:
                self._last_request_charge = float(
                    exc.headers.get("x-ms-request-charge", 0.0)
                )
            raise

    def _query_items(
        self,
        query: str,
        max_item_count: int = 100,
        partition_key: Optional[str] = None,
        metrics_recorder: Optional[RequestMetricsRecorder] = None,
        operation_name: str = "query_items",
    ) -> List[Dict[str, Any]]:
        kwargs = {"query": query, "max_item_count": max_item_count}
        if partition_key:
            kwargs["partition_key"] = partition_key
        else:
            kwargs["enable_cross_partition_query"] = True
        try:
            items = list(self._container.query_items(**kwargs))
            if metrics_recorder is not None:
                metrics_recorder.record_storage_read(
                    store="cosmos",
                    operation=operation_name,
                    item_count=len(items),
                    details={
                        "query": query.strip(),
                        "max_item_count": max_item_count,
                        "partition_key": partition_key,
                    },
                )
            return items
        except Exception as e:
            print(f"Cosmos query error: {e}")
            raise

    def create_session(
        self,
        user_id: str = "anonymous",
        title: Optional[str] = None,
        metrics_recorder: Optional[RequestMetricsRecorder] = None,
    ) -> str:
        session_id = str(uuid.uuid4())
        session_doc = {
            "id": f"session_{session_id}",
            "session_id": session_id,
            "user_id": user_id,
            "title": title,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message_count": 0,
            "doc_type": "session",
        }
        self._create_item(session_doc)
        self._capacity_monitor.register_cosmos_usage(
            request_charge=self._last_request_charge,
            storage_bytes=len(str(session_doc).encode("utf-8")),
        )
        self._capacity_monitor.increment_session_count()
        if metrics_recorder is not None:
            metrics_recorder.record_storage_write(
                store="cosmos",
                operation="create_session",
                request_charge=self._last_request_charge,
                bytes_written=len(str(session_doc).encode("utf-8")),
                details={"session_id": session_id, "user_id": user_id},
            )
        return session_id

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metrics_recorder: Optional[RequestMetricsRecorder] = None,
    ) -> Dict[str, Any]:
        message = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "doc_type": "message",
        }
        self._capacity_monitor.verify_cosmos_quota(message)
        self._create_item(message)
        self._capacity_monitor.register_cosmos_usage(
            request_charge=self._last_request_charge,
            storage_bytes=len(str(message).encode("utf-8")),
        )
        if metrics_recorder is not None:
            metrics_recorder.record_storage_write(
                store="cosmos",
                operation="save_message",
                request_charge=self._last_request_charge,
                bytes_written=len(str(message).encode("utf-8")),
                details={"session_id": session_id, "role": role},
            )
        self._update_session_message_count(
            session_id, 1, metrics_recorder=metrics_recorder
        )
        self._auto_set_title(session_id, role, content, metrics_recorder=metrics_recorder)
        return {"message": message, "request_charge": self._last_request_charge}

    def _auto_set_title(
        self,
        session_id: str,
        role: str,
        content: str,
        metrics_recorder: Optional[RequestMetricsRecorder] = None,
    ) -> None:
        if role != "user":
            return
        try:
            query = f"SELECT * FROM c WHERE c.session_id = '{session_id}' AND c.doc_type = 'session'"
            items = self._query_items(
                query,
                max_item_count=1,
                metrics_recorder=metrics_recorder,
                operation_name="auto_set_title_lookup",
            )
            if items:
                session_doc = items[0]
                if not session_doc.get("title"):
                    title = content[:50].replace("\n", " ").strip()
                    if len(content) > 50:
                        title += "..."
                    session_doc["title"] = title
                    self._container.upsert_item(session_doc)
                    if metrics_recorder is not None:
                        metrics_recorder.record_storage_write(
                            store="cosmos",
                            operation="auto_set_title_upsert",
                            details={"session_id": session_id, "title": title},
                        )
        except Exception:
            pass

    def _update_session_message_count(
        self,
        session_id: str,
        delta: int,
        metrics_recorder: Optional[RequestMetricsRecorder] = None,
    ) -> None:
        try:
            query = f"SELECT * FROM c WHERE c.session_id = '{session_id}' AND c.doc_type = 'session'"
            items = self._query_items(
                query,
                max_item_count=1,
                metrics_recorder=metrics_recorder,
                operation_name="update_message_count_lookup",
            )
            if items:
                session_doc = items[0]
                session_doc["message_count"] = (
                    session_doc.get("message_count", 0) + delta
                )
                self._container.upsert_item(session_doc)
                if metrics_recorder is not None:
                    metrics_recorder.record_storage_write(
                        store="cosmos",
                        operation="update_message_count_upsert",
                        details={"session_id": session_id, "delta": delta},
                    )
        except Exception:
            pass

    def get_session_history(
        self,
        session_id: str,
        limit: int = 100,
        metrics_recorder: Optional[RequestMetricsRecorder] = None,
    ) -> List[Dict[str, str]]:
        query = f"""
            SELECT c.id, c.role, c.content, c.timestamp
            FROM c
            WHERE c.session_id = '{session_id}' AND c.doc_type = 'message'
            ORDER BY c.timestamp ASC
        """
        try:
            items = self._query_items(
                query,
                max_item_count=limit,
                metrics_recorder=metrics_recorder,
                operation_name="get_session_history",
            )
            return [
                {"role": item["role"], "content": item["content"]} for item in items
            ]
        except CosmosHttpResponseError:
            return []

    def list_sessions(
        self, user_id: str = "anonymous", limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT c.session_id, c.title, c.created_at, c.message_count
            FROM c
            WHERE c.doc_type = 'session'
        """
        try:
            items = self._query_items(query, max_item_count=100)
            items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            items = items[offset : offset + limit]
            return [
                {
                    "session_id": item["session_id"],
                    "title": item.get("title")
                    or self._format_default_title(item.get("created_at", "")),
                    "created_at": item.get("created_at", ""),
                    "message_count": item.get("message_count", 0),
                }
                for item in items
            ]
        except CosmosHttpResponseError:
            return []

    def _format_default_title(self, created_at: str) -> str:
        if not created_at:
            return "Untitled Chat"
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            return dt.strftime("Chat - %b %d, %Y")
        except Exception:
            return "Untitled Chat"

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        query = f"SELECT * FROM c WHERE c.session_id = '{session_id}' AND c.doc_type = 'session'"
        try:
            items = self._query_items(query, max_item_count=1)
            if not items:
                return None
            item = items[0]
            return {
                "session_id": item["session_id"],
                "title": item.get("title")
                or self._format_default_title(item.get("created_at", "")),
                "created_at": item.get("created_at", ""),
                "message_count": item.get("message_count", 0),
            }
        except CosmosHttpResponseError:
            return None

    def update_session_title(self, session_id: str, title: str) -> bool:
        query = f"SELECT * FROM c WHERE c.session_id = '{session_id}' AND c.doc_type = 'session'"
        try:
            items = self._query_items(query, max_item_count=1)
            if not items:
                return False
            session_doc = items[0]
            session_doc["title"] = title.strip()[:100]
            self._container.upsert_item(session_doc)
            return True
        except CosmosHttpResponseError:
            return False

    def delete_session(self, session_id: str) -> None:
        query = f"SELECT c.id FROM c WHERE c.session_id = '{session_id}'"
        try:
            for item in self._query_items(query, max_item_count=100):
                self._container.delete_item(item=item["id"], partition_key=session_id)
        except CosmosHttpResponseError:
            pass

    def get_total_ru_charge(self) -> float:
        return self._capacity_monitor.usage.cosmos_rus_consumed


def get_chat_history_service() -> CosmosChatHistoryService:
    return CosmosChatHistoryService()
