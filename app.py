import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

import base64
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from services import (
    AzureQuotaExceededError,
    get_chat_history_service,
    get_document_store,
    get_llm_service,
    get_rag_cost_evaluator,
    get_speech_service,
)

load_dotenv()


@dataclass
class RuntimeConfig:
    default_index: str = "default-index"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_dimensions: int = 1536
    top_k: int = 4
    use_semantic_ranker: bool = False
    chat_temperature: float = 0.2
    chat_max_tokens: int = 1536
    use_vector_search: bool = True
    model: str = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
    stream_responses: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "azure_tier": os.getenv("AZURE_TIER", "FREE").upper(),
            "default_index": self.default_index,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_dimensions": self.embedding_dimensions,
            "top_k": self.top_k,
            "use_semantic_ranker": self.use_semantic_ranker,
            "chat_temperature": self.chat_temperature,
            "chat_max_tokens": self.chat_max_tokens,
            "use_vector_search": self.use_vector_search,
            "model": self.model,
            "stream_responses": self.stream_responses,
        }


runtime_config = RuntimeConfig()
chat_history: List[Dict[str, str]] = []

app = Flask(__name__)
document_store = get_document_store()
evaluator = get_rag_cost_evaluator()
llm_service = get_llm_service()


def _save_uploaded_file(uploaded_file) -> str:
    raw = uploaded_file.read()
    if not raw:
        return ""

    suffix = Path(uploaded_file.filename or "").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(raw)
        return handle.name


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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
        runtime_config.default_index = (
            str(payload["default_index"]).strip() or runtime_config.default_index
        )
    if "model" in payload:
        runtime_config.model = str(payload["model"]).strip() or runtime_config.model
    if "use_semantic_ranker" in payload:
        runtime_config.use_semantic_ranker = _coerce_bool(
            payload["use_semantic_ranker"], runtime_config.use_semantic_ranker
        )
    if "use_vector_search" in payload:
        runtime_config.use_vector_search = _coerce_bool(
            payload["use_vector_search"], runtime_config.use_vector_search
        )
    if "stream_responses" in payload:
        runtime_config.stream_responses = _coerce_bool(
            payload["stream_responses"], runtime_config.stream_responses
        )

    pricing = payload.get("pricing", {})
    if isinstance(pricing, dict):
        if "input_per_1k_tokens_usd" in pricing:
            os.environ["AZURE_LLM_INPUT_COST_PER_1K"] = str(
                pricing["input_per_1k_tokens_usd"]
            )
        if "output_per_1k_tokens_usd" in pricing:
            os.environ["AZURE_LLM_OUTPUT_COST_PER_1K"] = str(
                pricing["output_per_1k_tokens_usd"]
            )
        if "semantic_query_cost_usd" in pricing:
            os.environ["AZURE_SEMANTIC_QUERY_COST"] = str(
                pricing["semantic_query_cost_usd"]
            )
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


@app.get("/mic-test")
def mic_test() -> str:
    return render_template("mic-test.html")


@app.get("/secure-context-help")
def secure_context_help() -> str:
    return render_template("redirect.html")


@app.get("/settings")
def settings() -> str:
    return render_template("settings.html")


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
    index_name = (
        request.form.get("index_name", runtime_config.default_index).strip()
        or runtime_config.default_index
    )
    temp_path = ""
    try:
        temp_path = _save_uploaded_file(uploaded)
        if not temp_path:
            return jsonify({"ok": False, "error": "Uploaded file is empty."}), 400
        chunks, adi_pages = document_store.ingest_document(
            index_name=index_name,
            source_name=uploaded.filename or "uploaded_file",
            file_path=temp_path,
            chunk_size=runtime_config.chunk_size,
            overlap=runtime_config.chunk_overlap,
            vector_dimensions=runtime_config.embedding_dimensions,
        )
        if not chunks:
            return jsonify(
                {
                    "ok": False,
                    "error": "No valid text chunks were produced. File content appears non-text or extraction failed.",
                }
            ), 400
        embedding_bytes = len(chunks) * runtime_config.embedding_dimensions * 4
        cost_summary = evaluator.evaluate_ingestion(
            embedding_bytes_added=embedding_bytes,
            adi_pages_used=adi_pages,
        )
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
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


@app.get("/api/documents")
def list_documents() -> Any:
    index_name = (
        request.args.get("index_name", runtime_config.default_index).strip()
        or runtime_config.default_index
    )
    try:
        documents = document_store.list_documents(index_name=index_name)
        return jsonify({"ok": True, "index_name": index_name, "documents": documents})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.delete("/api/documents")
def delete_documents() -> Any:
    payload = request.get_json(silent=True) or {}
    index_name = (
        str(payload.get("index_name", runtime_config.default_index)).strip()
        or runtime_config.default_index
    )
    source_names = payload.get("source_names", [])
    if not isinstance(source_names, list) or not source_names:
        return jsonify({"ok": False, "error": "source_names list is required"}), 400

    try:
        deleted = document_store.delete_by_sources(
            index_name=index_name, source_names=source_names
        )
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
    import logging

    logger = logging.getLogger(__name__)
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    if not query:
        return jsonify({"ok": False, "error": "query is required"}), 400

    index_name = (
        str(payload.get("index_name", runtime_config.default_index)).strip()
        or runtime_config.default_index
    )
    top_k = int(payload.get("top_k", runtime_config.top_k))
    use_semantic_ranker = _coerce_bool(
        payload.get("use_semantic_ranker"), runtime_config.use_semantic_ranker
    )
    use_vector_search = _coerce_bool(
        payload.get("use_vector_search"), runtime_config.use_vector_search
    )
    temperature = float(payload.get("temperature", runtime_config.chat_temperature))
    max_tokens = int(payload.get("max_tokens", runtime_config.chat_max_tokens))
    model = str(payload.get("model", runtime_config.model))
    session_id = payload.get("session_id")

    cosmos_service = get_chat_history_service()
    if not session_id:
        session_id = cosmos_service.create_session(user_id="anonymous")
        logger.info(f"Created new session: {session_id}")

    cosmos_history = cosmos_service.get_session_history(session_id)

    contexts = document_store.search(
        query=query,
        index_name=index_name,
        top_k=top_k,
        use_semantic_ranker=use_semantic_ranker,
        use_vector_search=use_vector_search,
        vector_dimensions=runtime_config.embedding_dimensions,
    )
    context_texts = [c.text for c in contexts]
    context_payload = [
        {"chunk_id": c.chunk_id, "source_name": c.source_name, "text": c.text}
        for c in contexts
    ]

    try:
        llm_result = llm_service.generate_rag_answer(
            query=query,
            contexts=context_texts,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            chat_history=cosmos_history,
        )
        cost_summary = evaluator.evaluate_query(
            input_tokens=int(llm_result["input_tokens"]),
            output_tokens=int(llm_result["output_tokens"]),
            use_semantic_ranker=use_semantic_ranker,
        )

        cosmos_service.save_message(session_id, "user", query)
        cosmos_service.save_message(session_id, "assistant", str(llm_result["answer"]))

        chat_history.clear()
        chat_history.extend(cosmos_history[-20:])

        logger.info(
            f"Chat response: session={session_id}, tokens={llm_result['input_tokens']}+{llm_result['output_tokens']}, "
            f"contexts={len(contexts)}, cosmos_rus={cost_summary.get('cosmos_rus_consumed', 0)}, "
            f"total_cost=${cost_summary.get('total_session_cost_usd', 0):.6f}"
        )

        return jsonify(
            {
                "ok": True,
                "session_id": session_id,
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
        logger.error(f"Quota exceeded: {exc}")
        return jsonify({"ok": False, "error": str(exc)}), 429
    except Exception as exc:
        logger.error(f"Chat error: {exc}", exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/chat/stream")
def chat_stream() -> Any:
    from flask import Response, stream_with_context
    import json
    import logging

    logger = logging.getLogger(__name__)

    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    if not query:
        return jsonify({"ok": False, "error": "query is required"}), 400

    index_name = (
        str(payload.get("index_name", runtime_config.default_index)).strip()
        or runtime_config.default_index
    )
    top_k = int(payload.get("top_k", runtime_config.top_k))
    use_semantic_ranker = _coerce_bool(
        payload.get("use_semantic_ranker"), runtime_config.use_semantic_ranker
    )
    use_vector_search = _coerce_bool(
        payload.get("use_vector_search"), runtime_config.use_vector_search
    )
    temperature = float(payload.get("temperature", runtime_config.chat_temperature))
    max_tokens = int(payload.get("max_tokens", runtime_config.chat_max_tokens))
    model = str(payload.get("model", runtime_config.model))
    session_id = payload.get("session_id")

    cosmos_service = get_chat_history_service()
    if not session_id:
        session_id = cosmos_service.create_session(user_id="anonymous")

    cosmos_history = cosmos_service.get_session_history(session_id)

    contexts = document_store.search(
        query=query,
        index_name=index_name,
        top_k=top_k,
        use_semantic_ranker=use_semantic_ranker,
        use_vector_search=use_vector_search,
        vector_dimensions=runtime_config.embedding_dimensions,
    )
    context_texts = [c.text for c in contexts]
    context_payload = [
        {"chunk_id": c.chunk_id, "source_name": c.source_name, "text": c.text}
        for c in contexts
    ]

    def generate():
        try:
            stream_gen = llm_service.stream_rag_answer(
                query=query,
                contexts=context_texts,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=cosmos_history,
            )

            full_answer = ""
            token_count = 0
            for chunk in stream_gen:
                full_answer += chunk
                token_count += 1
                yield f"data: {json.dumps({'type': 'token', 'token': chunk})}\n\n"

            meta = stream_gen.send(None) if hasattr(stream_gen, "send") else {}
            if not isinstance(meta, dict):
                meta = {}

            cost_summary = evaluator.evaluate_query(
                input_tokens=meta.get(
                    "input_tokens",
                    max(int(len(" ".join(context_texts).split()) * 1.3), 1),
                ),
                output_tokens=meta.get("output_tokens", token_count),
                use_semantic_ranker=use_semantic_ranker,
            )

            cosmos_service.save_message(session_id, "user", query)
            cosmos_service.save_message(session_id, "assistant", full_answer)

            chat_history.clear()
            chat_history.extend(cosmos_history[-20:])

            final_data = {
                "type": "done",
                "session_id": session_id,
                "answer": full_answer,
                "model": meta.get("model", model),
                "contexts": context_payload,
                "usage": {
                    "input_tokens": meta.get("input_tokens", 0),
                    "output_tokens": meta.get("output_tokens", 0),
                },
                "cost_summary": cost_summary,
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            logger.info(f"Stream chat: session={session_id}, tokens={token_count}")
        except Exception as exc:
            logger.error(f"Stream chat error: {exc}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/voice-chat")
def voice_chat() -> Any:
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "audio file is required"}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"ok": False, "error": "filename is required"}), 400

    temp_audio_path = ""
    try:
        suffix = Path(uploaded_file.filename or "").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(uploaded_file.read())
            temp_audio_path = handle.name

        if not temp_audio_path or not os.path.exists(temp_audio_path):
            return jsonify({"ok": False, "error": "Failed to save audio file."}), 500

        speech_service = get_speech_service()
        query = speech_service.transcribe_audio(temp_audio_path)

        index_name = (
            request.form.get("index_name", runtime_config.default_index).strip()
            or runtime_config.default_index
        )
        top_k = int(request.form.get("top_k", runtime_config.top_k))
        use_semantic_ranker = _coerce_bool(
            request.form.get("use_semantic_ranker"), runtime_config.use_semantic_ranker
        )
        use_vector_search = _coerce_bool(
            request.form.get("use_vector_search"), runtime_config.use_vector_search
        )
        temperature = float(
            request.form.get("temperature", runtime_config.chat_temperature)
        )
        max_tokens = int(request.form.get("max_tokens", runtime_config.chat_max_tokens))
        model = str(request.form.get("model", runtime_config.model))
        session_id = request.form.get("session_id")

        cosmos_service = get_chat_history_service()
        if not session_id:
            session_id = cosmos_service.create_session(user_id="anonymous")

        cosmos_history = cosmos_service.get_session_history(session_id)

        contexts = document_store.search(
            query=query,
            index_name=index_name,
            top_k=top_k,
            use_semantic_ranker=use_semantic_ranker,
            use_vector_search=use_vector_search,
            vector_dimensions=runtime_config.embedding_dimensions,
        )
        context_texts = [c.text for c in contexts]
        context_payload = [
            {"chunk_id": c.chunk_id, "source_name": c.source_name, "text": c.text}
            for c in contexts
        ]

        llm_result = llm_service.generate_rag_answer(
            query=query,
            contexts=context_texts,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            chat_history=cosmos_history,
        )
        cost_summary = evaluator.evaluate_query(
            input_tokens=int(llm_result["input_tokens"]),
            output_tokens=int(llm_result["output_tokens"]),
            use_semantic_ranker=use_semantic_ranker,
        )

        cosmos_service.save_message(session_id, "user", query)
        cosmos_service.save_message(session_id, "assistant", str(llm_result["answer"]))

        chat_history.clear()
        chat_history.extend(cosmos_history[-20:])

        answer_text = str(llm_result["answer"])
        tts_audio_data = speech_service.synthesize_speech(answer_text)
        tts_audio_base64 = base64.b64encode(tts_audio_data).decode("utf-8")

        return jsonify(
            {
                "ok": True,
                "session_id": session_id,
                "answer": answer_text,
                "model": llm_result["model"],
                "contexts": context_payload,
                "usage": {
                    "input_tokens": llm_result["input_tokens"],
                    "output_tokens": llm_result["output_tokens"],
                },
                "audio": tts_audio_base64,
                "cost_summary": cost_summary,
            }
        )
    except AzureQuotaExceededError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 429
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError:
                pass


@app.post("/api/chat-with-tts")
def chat_with_tts() -> Any:
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    if not query:
        return jsonify({"ok": False, "error": "query is required"}), 400

    index_name = (
        str(payload.get("index_name", runtime_config.default_index)).strip()
        or runtime_config.default_index
    )
    top_k = int(payload.get("top_k", runtime_config.top_k))
    use_semantic_ranker = _coerce_bool(
        payload.get("use_semantic_ranker"), runtime_config.use_semantic_ranker
    )
    use_vector_search = _coerce_bool(
        payload.get("use_vector_search"), runtime_config.use_vector_search
    )
    temperature = float(payload.get("temperature", runtime_config.chat_temperature))
    max_tokens = int(payload.get("max_tokens", runtime_config.chat_max_tokens))
    model = str(payload.get("model", runtime_config.model))
    session_id = payload.get("session_id")

    cosmos_service = get_chat_history_service()
    if not session_id:
        session_id = cosmos_service.create_session(user_id="anonymous")

    cosmos_history = cosmos_service.get_session_history(session_id)

    contexts = document_store.search(
        query=query,
        index_name=index_name,
        top_k=top_k,
        use_semantic_ranker=use_semantic_ranker,
        use_vector_search=use_vector_search,
        vector_dimensions=runtime_config.embedding_dimensions,
    )
    context_texts = [c.text for c in contexts]
    context_payload = [
        {"chunk_id": c.chunk_id, "source_name": c.source_name, "text": c.text}
        for c in contexts
    ]

    try:
        llm_result = llm_service.generate_rag_answer(
            query=query,
            contexts=context_texts,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            chat_history=cosmos_history,
        )
        cost_summary = evaluator.evaluate_query(
            input_tokens=int(llm_result["input_tokens"]),
            output_tokens=int(llm_result["output_tokens"]),
            use_semantic_ranker=use_semantic_ranker,
        )

        cosmos_service.save_message(session_id, "user", query)
        cosmos_service.save_message(session_id, "assistant", str(llm_result["answer"]))

        chat_history.clear()
        chat_history.extend(cosmos_history[-20:])

        answer_text = str(llm_result["answer"])

        speech_service = get_speech_service()
        tts_audio_data = speech_service.synthesize_speech(answer_text)
        tts_audio_base64 = base64.b64encode(tts_audio_data).decode("utf-8")

        return jsonify(
            {
                "ok": True,
                "session_id": session_id,
                "answer": answer_text,
                "model": llm_result["model"],
                "contexts": context_payload,
                "usage": {
                    "input_tokens": llm_result["input_tokens"],
                    "output_tokens": llm_result["output_tokens"],
                },
                "audio": tts_audio_base64,
                "cost_summary": cost_summary,
            }
        )
    except AzureQuotaExceededError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 429
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/reset")
def reset_session() -> Any:
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    if session_id:
        try:
            cosmos_service = get_chat_history_service()
            cosmos_service.delete_session(session_id)
        except Exception:
            pass
    chat_history.clear()
    evaluator.capacity_monitor.reset_usage()
    return jsonify({"ok": True, "message": "Session usage and chat history reset"})


@app.get("/api/tts-config")
def get_tts_config() -> Any:
    try:
        speech_service = get_speech_service()
        settings = speech_service.get_tts_settings()
        return jsonify({"ok": True, "settings": settings})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/tts-config")
def set_tts_config() -> Any:
    payload = request.get_json(silent=True) or {}
    try:
        speech_service = get_speech_service()

        if "voice" in payload:
            speech_service.update_tts_settings(voice=payload["voice"])
        if "speed" in payload:
            speed = float(payload["speed"])
            speech_service.update_tts_settings(speed=speed)
        if "pitch" in payload:
            pitch = int(payload["pitch"])
            speech_service.update_tts_settings(pitch=pitch)
        if "style" in payload:
            speech_service.update_tts_settings(style=payload["style"])

        settings = speech_service.get_tts_settings()
        return jsonify({"ok": True, "settings": settings})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/tts-test")
def test_tts() -> Any:
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "Hello! This is a test of the text to speech system.")
    try:
        speech_service = get_speech_service()
        audio_data = speech_service.synthesize_speech(text)
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        return jsonify({"ok": True, "audio": audio_base64})
    except AzureQuotaExceededError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 429
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/tts-generate")
def generate_tts() -> Any:
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    if not text:
        return jsonify({"ok": False, "error": "text is required"}), 400
    try:
        speech_service = get_speech_service()
        audio_data = speech_service.synthesize_speech(text)
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        return jsonify({"ok": True, "audio": audio_base64})
    except AzureQuotaExceededError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 429
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/sessions")
def list_sessions() -> Any:
    import logging

    logger = logging.getLogger(__name__)
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))
    try:
        cosmos_service = get_chat_history_service()
        sessions = cosmos_service.list_sessions(limit=limit, offset=offset)
        logger.info(f"List sessions: found {len(sessions)} sessions")
        return jsonify({"ok": True, "sessions": sessions})
    except Exception as exc:
        logger.error(f"List sessions error: {exc}", exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/sessions/<session_id>")
def get_session(session_id: str) -> Any:
    import logging

    logger = logging.getLogger(__name__)
    try:
        cosmos_service = get_chat_history_service()
        info = cosmos_service.get_session_info(session_id)
        if not info:
            logger.warning(f"Session not found: {session_id}")
            return jsonify({"ok": False, "error": "Session not found"}), 404
        messages = cosmos_service.get_session_history(session_id, limit=100)
        logger.info(f"Get session {session_id}: {len(messages)} messages")
        return jsonify({"ok": True, "session": info, "messages": messages})
    except Exception as exc:
        logger.error(f"Get session error: {exc}", exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.put("/api/sessions/<session_id>")
def update_session(session_id: str) -> Any:
    import logging

    logger = logging.getLogger(__name__)
    payload = request.get_json(silent=True) or {}
    title = payload.get("title", "")
    if not title:
        return jsonify({"ok": False, "error": "title is required"}), 400
    try:
        cosmos_service = get_chat_history_service()
        success = cosmos_service.update_session_title(session_id, title)
        if not success:
            return jsonify({"ok": False, "error": "Session not found"}), 404
        logger.info(f"Renamed session {session_id} to: {title}")
        return jsonify({"ok": True})
    except Exception as exc:
        logger.error(f"Update session error: {exc}", exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.delete("/api/sessions/<session_id>")
def delete_session_endpoint(session_id: str) -> Any:
    import logging

    logger = logging.getLogger(__name__)
    try:
        cosmos_service = get_chat_history_service()
        cosmos_service.delete_session(session_id)
        logger.info(f"Deleted session: {session_id}")
        return jsonify({"ok": True})
    except Exception as exc:
        logger.error(f"Delete session error: {exc}", exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
