import os
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from services import (
    AzureQuotaExceededError,
    get_document_store,
    get_llm_service,
    get_rag_cost_evaluator,
)

load_dotenv()


@dataclass
class RuntimeConfig:
    default_index: str = "default-index"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_dimensions: int = 1536
    top_k: int = 4
    chat_temperature: float = 0.2
    chat_max_tokens: int = 600
    use_vector_search: bool = True
    model: str = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "azure_tier": os.getenv("AZURE_TIER", "FREE").upper(),
            "default_index": self.default_index,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_dimensions": self.embedding_dimensions,
            "top_k": self.top_k,
            "chat_temperature": self.chat_temperature,
            "chat_max_tokens": self.chat_max_tokens,
            "use_vector_search": self.use_vector_search,
            "model": self.model,
            "pricing": {
                "input_per_1k_tokens_usd": float(os.getenv("AZURE_LLM_INPUT_COST_PER_1K", "0.00015")),
                "output_per_1k_tokens_usd": float(os.getenv("AZURE_LLM_OUTPUT_COST_PER_1K", "0.00060")),
                "semantic_query_cost_usd": float(os.getenv("AZURE_SEMANTIC_QUERY_COST", "0.001")),
            },
        }


runtime_config = RuntimeConfig()
chat_history: List[Dict[str, str]] = []

app = Flask(__name__)
document_store = get_document_store()
evaluator = get_rag_cost_evaluator()
llm_service = get_llm_service()


def _read_uploaded_file(uploaded_file) -> str:
    raw = uploaded_file.read()
    if not raw:
        return ""
    return raw.decode("utf-8", errors="ignore")


def _update_config(payload: Dict[str, Any]) -> None:
    global runtime_config
    if "azure_tier" in payload:
        tier = str(payload["azure_tier"]).strip().upper()
        if tier not in {"FREE", "BASIC"}:
            raise ValueError("azure_tier must be FREE or BASIC")
        os.environ["AZURE_TIER"] = tier
        evaluator.capacity_monitor.refresh_tier()

    int_fields = {
        "chunk_size": "chunk_size",
        "chunk_overlap": "chunk_overlap",
        "embedding_dimensions": "embedding_dimensions",
        "top_k": "top_k",
        "chat_max_tokens": "chat_max_tokens",
    }
    for k, attr in int_fields.items():
        if k in payload:
            setattr(runtime_config, attr, int(payload[k]))

    if "chat_temperature" in payload:
        runtime_config.chat_temperature = float(payload["chat_temperature"])
    if "default_index" in payload:
        runtime_config.default_index = str(payload["default_index"]).strip() or runtime_config.default_index
    if "model" in payload:
        runtime_config.model = str(payload["model"]).strip() or runtime_config.model
    if "use_vector_search" in payload:
        runtime_config.use_vector_search = bool(payload["use_vector_search"])

    pricing = payload.get("pricing", {})
    if isinstance(pricing, dict):
        if "input_per_1k_tokens_usd" in pricing:
            os.environ["AZURE_LLM_INPUT_COST_PER_1K"] = str(pricing["input_per_1k_tokens_usd"])
        if "output_per_1k_tokens_usd" in pricing:
            os.environ["AZURE_LLM_OUTPUT_COST_PER_1K"] = str(pricing["output_per_1k_tokens_usd"])
        if "semantic_query_cost_usd" in pricing:
            os.environ["AZURE_SEMANTIC_QUERY_COST"] = str(pricing["semantic_query_cost_usd"])
        evaluator.pricing.input_per_1k_tokens_usd = float(
            os.getenv("AZURE_LLM_INPUT_COST_PER_1K", "0.00015")
        )
        evaluator.pricing.output_per_1k_tokens_usd = float(
            os.getenv("AZURE_LLM_OUTPUT_COST_PER_1K", "0.00060")
        )
        evaluator.pricing.semantic_ranker_per_query_usd = float(
            os.getenv("AZURE_SEMANTIC_QUERY_COST", "0.001")
        )


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/health")
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "config": runtime_config.to_dict(),
            "store": document_store.stats(),
        }
    )


@app.get("/api/config")
def get_config() -> Any:
    return jsonify(runtime_config.to_dict())


@app.post("/api/config")
def set_config() -> Any:
    payload = request.get_json(silent=True) or {}
    try:
        _update_config(payload)
        return jsonify({"ok": True, "config": runtime_config.to_dict()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/upload")
def upload_document() -> Any:
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "file is required"}), 400

    uploaded = request.files["file"]
    index_name = request.form.get("index_name", runtime_config.default_index).strip() or runtime_config.default_index
    try:
        text = _read_uploaded_file(uploaded)
        chunks = document_store.ingest_text(
            index_name=index_name,
            source_name=uploaded.filename or "uploaded_file",
            text=text,
            chunk_size=runtime_config.chunk_size,
            overlap=runtime_config.chunk_overlap,
            vector_dimensions=runtime_config.embedding_dimensions,
        )
        embedding_bytes = len(chunks) * runtime_config.embedding_dimensions * 4
        cost_summary = evaluator.evaluate_ingestion(embedding_bytes_added=embedding_bytes)
        return jsonify(
            {
                "ok": True,
                "index_name": index_name,
                "file_name": uploaded.filename,
                "chunks_ingested": len(chunks),
                "cost_summary": cost_summary,
                "store": document_store.stats(),
            }
        )
    except AzureQuotaExceededError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 429
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/documents")
def list_documents() -> Any:
    index_name = request.args.get("index_name", runtime_config.default_index).strip() or runtime_config.default_index
    try:
        documents = document_store.list_documents(index_name=index_name)
        return jsonify({"ok": True, "index_name": index_name, "documents": documents})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.delete("/api/documents")
def delete_documents() -> Any:
    payload = request.get_json(silent=True) or {}
    index_name = str(payload.get("index_name", runtime_config.default_index)).strip() or runtime_config.default_index
    source_names = payload.get("source_names", [])
    if not isinstance(source_names, list) or not source_names:
        return jsonify({"ok": False, "error": "source_names list is required"}), 400

    try:
        deleted = document_store.delete_by_sources(index_name=index_name, source_names=source_names)
        return jsonify(
            {
                "ok": True,
                "index_name": index_name,
                "delete_summary": deleted,
                "store": document_store.stats(),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/chat")
def chat() -> Any:
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    if not query:
        return jsonify({"ok": False, "error": "query is required"}), 400

    index_name = str(payload.get("index_name", runtime_config.default_index)).strip() or runtime_config.default_index
    top_k = int(payload.get("top_k", runtime_config.top_k))
    use_semantic_ranker = bool(payload.get("use_semantic_ranker", False))
    use_vector_search = bool(payload.get("use_vector_search", runtime_config.use_vector_search))
    temperature = float(payload.get("temperature", runtime_config.chat_temperature))
    max_tokens = int(payload.get("max_tokens", runtime_config.chat_max_tokens))
    model = str(payload.get("model", runtime_config.model))

    contexts = document_store.search(
        query=query,
        index_name=index_name,
        top_k=top_k,
        use_semantic_ranker=use_semantic_ranker,
        use_vector_search=use_vector_search,
        vector_dimensions=runtime_config.embedding_dimensions,
    )
    context_texts = [c.text for c in contexts]
    context_payload = [{"chunk_id": c.chunk_id, "source_name": c.source_name, "text": c.text} for c in contexts]

    try:
        llm_result = llm_service.generate_rag_answer(
            query=query,
            contexts=context_texts,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            chat_history=chat_history,
        )
        cost_summary = evaluator.evaluate_query(
            input_tokens=int(llm_result["input_tokens"]),
            output_tokens=int(llm_result["output_tokens"]),
            use_semantic_ranker=use_semantic_ranker,
        )
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": str(llm_result["answer"])})
        if len(chat_history) > 20:
            del chat_history[:-20]

        return jsonify(
            {
                "ok": True,
                "answer": llm_result["answer"],
                "model": llm_result["model"],
                "contexts": context_payload,
                "usage": {
                    "input_tokens": llm_result["input_tokens"],
                    "output_tokens": llm_result["output_tokens"],
                },
                "cost_summary": cost_summary,
            }
        )
    except AzureQuotaExceededError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 429
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/reset")
def reset_session() -> Any:
    chat_history.clear()
    evaluator.capacity_monitor.reset_usage()
    return jsonify({"ok": True, "message": "Session usage and chat history reset"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
