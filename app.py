import base64
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from services.request_metrics import (
    DocumentUploadMetricsRecorder,
    RequestMetricsRecorder,
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
REQUEST_METRICS_DIR = Path("request_metrics")
SPEECH_STT_COST_PER_HOUR_USD = 1.00
SPEECH_TTS_COST_PER_MILLION_CHARS_USD = 15.00
COSMOS_REQUEST_UNIT_COST_PER_100_UNITS_USD = 0.0
AZURE_SEARCH_BASIC_PRICE_PER_SU_MONTH_INR = 6706.12
DEFAULT_AZURE_SEARCH_UNITS = 1
SECONDS_PER_30_DAY_MONTH = 30 * 24 * 60 * 60
SPEECH_STT_REALTIME_PRICE_PER_HOUR_INR = 90.956
SPEECH_TTS_NEURAL_PRICE_PER_MILLION_CHARS_INR = 1364.326
LLM_PRICING_OPTIONS: List[Dict[str, Any]] = [
    {
        "id": "openai/gpt-oss-20b",
        "label": "GPT-OSS 20B",
        "input_per_1k_tokens_usd": 0.00015,
        "output_per_1k_tokens_usd": 0.00060,
    },
    {
        "id": "openai/gpt-oss-120b",
        "label": "GPT-OSS 120B",
        "input_per_1k_tokens_usd": 0.0009,
        "output_per_1k_tokens_usd": 0.0036,
    },
    {
        "id": "meta/llama-3.1-8b-instruct",
        "label": "Llama 3.1 8B Instruct",
        "input_per_1k_tokens_usd": 0.00012,
        "output_per_1k_tokens_usd": 0.00024,
    },
    {
        "id": "meta/llama-3.3-70b-instruct",
        "label": "Llama 3.3 70B Instruct",
        "input_per_1k_tokens_usd": 0.00059,
        "output_per_1k_tokens_usd": 0.00079,
    },
    {
    "id": "openai/gpt-5.4",
    "label": "GPT-5.4",
    "input_per_1k_tokens_usd": 0.0025,
    "output_per_1k_tokens_usd": 0.015,
    },
    {
        "id": "openai/gpt-5.4-mini",
        "label": "GPT-5.4 Mini",
        "input_per_1k_tokens_usd": 0.00075,
        "output_per_1k_tokens_usd": 0.0045,
    },
    {
        "id": "openai/gpt-5.4-nano",
        "label": "GPT-5.4 Nano",
        "input_per_1k_tokens_usd": 0.0002,
        "output_per_1k_tokens_usd": 0.00125,
    },
    {
        "id": "openai/gpt-5",
        "label": "GPT-5",
        "input_per_1k_tokens_usd": 0.00125,
        "output_per_1k_tokens_usd": 0.01,
    },
    {
        "id": "openai/gpt-5-mini",
        "label": "GPT-5 Mini",
        "input_per_1k_tokens_usd": 0.00025,
        "output_per_1k_tokens_usd": 0.002,
    },
    {
        "id": "openai/gpt-4o",
        "label": "GPT-4o",
        "input_per_1k_tokens_usd": 0.0025,
        "output_per_1k_tokens_usd": 0.01,
    },
    {
        "id": "openai/gpt-4o-mini",
        "label": "GPT-4o Mini",
        "input_per_1k_tokens_usd": 0.00015,
        "output_per_1k_tokens_usd": 0.0006,
    },
    {
        "id": "google/gemini-3.1-pro-preview",
        "label": "Gemini 3.1 Pro Preview",
        "input_per_1k_tokens_usd": 0.002,
        "output_per_1k_tokens_usd": 0.012,
    },
    {
        "id": "google/gemini-3-flash",
        "label": "Gemini 3 Flash",
        "input_per_1k_tokens_usd": 0.0005,
        "output_per_1k_tokens_usd": 0.003,
    },
    {
        "id": "google/gemini-2.5-pro",
        "label": "Gemini 2.5 Pro",
        "input_per_1k_tokens_usd": 0.00125,
        "output_per_1k_tokens_usd": 0.01,
    },
    {
        "id": "google/gemini-2.5-flash",
        "label": "Gemini 2.5 Flash",
        "input_per_1k_tokens_usd": 0.0003,
        "output_per_1k_tokens_usd": 0.0025,
    },
    {
        "id": "google/gemini-2.5-flash-lite",
        "label": "Gemini 2.5 Flash-Lite",
        "input_per_1k_tokens_usd": 0.0001,
        "output_per_1k_tokens_usd": 0.0004,
    },
    {
        "id": "anthropic/claude-opus-4.6",
        "label": "Claude Opus 4.6",
        "input_per_1k_tokens_usd": 0.005,
        "output_per_1k_tokens_usd": 0.025,
    },
    {
        "id": "anthropic/claude-sonnet-4.6",
        "label": "Claude Sonnet 4.6",
        "input_per_1k_tokens_usd": 0.003,
        "output_per_1k_tokens_usd": 0.015,
    },
    {
        "id": "anthropic/claude-haiku-4.5",
        "label": "Claude Haiku 4.5",
        "input_per_1k_tokens_usd": 0.001,
        "output_per_1k_tokens_usd": 0.005,
    }
]

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


def _metrics_file_value(metrics_path: Path) -> str:
    return metrics_path.as_posix()


def _estimate_adi_cost_usd(pages: int) -> float:
    if str(os.getenv("AZURE_TIER", "FREE")).strip().upper() != "BASIC":
        return 0.0
    return round((max(int(pages), 0) / 1000.0) * 10.0, 6)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_metrics_label(metrics: Dict[str, Any], fallback: str) -> str:
    query = str(metrics.get("user_query") or "").strip()
    return query or fallback


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _load_metrics_payload(metrics_name: str) -> Dict[str, Any]:
    safe_name = Path(metrics_name).name
    metrics_path = REQUEST_METRICS_DIR / safe_name
    if not metrics_path.exists() or metrics_path.suffix.lower() != ".json":
        raise FileNotFoundError(f"Metrics file '{safe_name}' was not found.")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _list_request_metrics() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not REQUEST_METRICS_DIR.exists():
        return items

    for metrics_path in sorted(REQUEST_METRICS_DIR.glob("*.json"), reverse=True):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict) or "llm" not in payload:
            continue
        llm_payload = payload.get("llm") or {}
        items.append(
            {
                "file_name": metrics_path.name,
                "label": _format_metrics_label(payload, metrics_path.name),
                "user_query": payload.get("user_query"),
                "recorded_model": llm_payload.get("model"),
                "input_tokens": _safe_int(llm_payload.get("input_tokens")),
                "output_tokens": _safe_int(llm_payload.get("output_tokens")),
                "status": payload.get("status"),
                "started_at": payload.get("started_at"),
                "finished_at": payload.get("finished_at"),
            }
        )
    return items


def _get_llm_pricing_options() -> List[Dict[str, Any]]:
    options = [dict(item) for item in LLM_PRICING_OPTIONS]
    known_ids = {item["id"] for item in options}
    runtime_model_id = str(runtime_config.model or "").strip()
    if runtime_model_id and runtime_model_id not in known_ids:
        options.insert(
            0,
            {
                "id": runtime_model_id,
                "label": f"{runtime_model_id} (Configured Runtime)",
                "input_per_1k_tokens_usd": _safe_float(
                    os.getenv("AZURE_LLM_INPUT_COST_PER_1K", "0.00015"), 0.00015
                ),
                "output_per_1k_tokens_usd": _safe_float(
                    os.getenv("AZURE_LLM_OUTPUT_COST_PER_1K", "0.00060"), 0.00060
                ),
            },
        )
    return options


def _resolve_llm_pricing(model_id: str) -> Dict[str, Any]:
    selected_id = str(model_id or "").strip()
    for option in _get_llm_pricing_options():
        if option["id"] == selected_id:
            return option
    raise ValueError(f"Unsupported model '{selected_id}'.")


def _estimate_request_cost(metrics: Dict[str, Any], selected_model_id: str) -> Dict[str, Any]:
    llm_payload = metrics.get("llm") or {}
    vector_store = metrics.get("vector_store") or {}
    speech = metrics.get("speech") or {}
    storage = metrics.get("storage") or {}
    stt = speech.get("stt") or {}
    tts = speech.get("tts") or {}
    tier = str(os.getenv("AZURE_TIER", "FREE")).strip().upper()
    selected_pricing = _resolve_llm_pricing(selected_model_id)
    started_at = _parse_iso_datetime(metrics.get("started_at"))
    finished_at = _parse_iso_datetime(metrics.get("finished_at"))

    input_tokens = _safe_int(llm_payload.get("input_tokens"))
    output_tokens = _safe_int(llm_payload.get("output_tokens"))
    semantic_queries = _safe_int(vector_store.get("semantic_query_count"))
    used_semantic_ranker = bool(vector_store.get("used_semantic_ranker"))
    result_chunks = vector_store.get("result_chunks") or []
    retrieved_text_length = sum(
        _safe_int(chunk.get("text_length"))
        for chunk in result_chunks
        if isinstance(chunk, dict)
    )
    request_duration_seconds = 0.0
    if started_at and finished_at:
        request_duration_seconds = max(
            (finished_at - started_at).total_seconds(),
            0.0,
        )

    llm_input_cost = (
        input_tokens / 1000.0
    ) * _safe_float(selected_pricing.get("input_per_1k_tokens_usd"))
    llm_output_cost = (
        output_tokens / 1000.0
    ) * _safe_float(selected_pricing.get("output_per_1k_tokens_usd"))
    llm_cost = llm_input_cost + llm_output_cost

    semantic_unit_cost = _safe_float(os.getenv("AZURE_SEMANTIC_QUERY_COST", "0.001"), 0.001)
    semantic_cost = semantic_queries * semantic_unit_cost if used_semantic_ranker else 0.0
    azure_search_compute_cost_inr = (
        (
            AZURE_SEARCH_BASIC_PRICE_PER_SU_MONTH_INR
            * DEFAULT_AZURE_SEARCH_UNITS
        )
        / float(SECONDS_PER_30_DAY_MONTH)
    ) * request_duration_seconds

    stt_seconds = _safe_float(stt.get("input_audio_seconds"))
    tts_characters = _safe_int(tts.get("input_characters"))
    stt_cost = (
        (stt_seconds / 3600.0) * SPEECH_STT_COST_PER_HOUR_USD if tier == "BASIC" else 0.0
    )
    tts_cost = (
        (tts_characters / 1_000_000.0) * SPEECH_TTS_COST_PER_MILLION_CHARS_USD
        if tier == "BASIC"
        else 0.0
    )
    stt_cost_inr = (
        (stt_seconds / 3600.0) * SPEECH_STT_REALTIME_PRICE_PER_HOUR_INR
    )
    tts_cost_inr = (
        (tts_characters / 1_000_000.0) * SPEECH_TTS_NEURAL_PRICE_PER_MILLION_CHARS_INR
    )
    speech_cost = stt_cost + tts_cost
    speech_cost_inr = stt_cost_inr + tts_cost_inr

    read_operations = storage.get("read_operations") or []
    write_operations = storage.get("write_operations") or []
    cosmos_request_units = 0.0
    cosmos_bytes_written = 0
    for operation in [*read_operations, *write_operations]:
        if operation.get("store") != "cosmos":
            continue
        cosmos_request_units += _safe_float(operation.get("request_charge"))
        cosmos_bytes_written += _safe_int(operation.get("bytes_written"))
    cosmos_cost = (
        (cosmos_request_units / 100.0) * COSMOS_REQUEST_UNIT_COST_PER_100_UNITS_USD
        if cosmos_request_units > 0
        else 0.0
    )

    total_cost = llm_cost + semantic_cost + speech_cost + cosmos_cost
    return {
        "metrics_file": metrics.get("_file_name"),
        "user_query": metrics.get("user_query"),
        "status": metrics.get("status"),
        "recorded_model": llm_payload.get("model"),
        "selected_model": selected_pricing["id"],
        "pricing_override_applied": selected_pricing["id"] != llm_payload.get("model"),
        "started_at": metrics.get("started_at"),
        "finished_at": metrics.get("finished_at"),
        "request_duration_seconds": round(request_duration_seconds, 3),
        "total_cost_usd": round(total_cost, 6),
        "total_cost_inr": round(azure_search_compute_cost_inr + speech_cost_inr, 6),
        "services": [
            {
                "key": "llm",
                "label": "LLM",
                "cost_usd": round(llm_cost, 6),
                "details": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_cost_usd": round(llm_input_cost, 6),
                    "output_cost_usd": round(llm_output_cost, 6),
                    "input_per_1k_tokens_usd": _safe_float(
                        selected_pricing.get("input_per_1k_tokens_usd")
                    ),
                    "output_per_1k_tokens_usd": _safe_float(
                        selected_pricing.get("output_per_1k_tokens_usd")
                    ),
                    "priced_model": selected_pricing["id"],
                    "recorded_model": llm_payload.get("model"),
                },
            },
            {
                "key": "azure_search",
                "label": "Azure AI Search",
                "cost_usd": round(semantic_cost, 6),
                "cost_inr": round(azure_search_compute_cost_inr, 6),
                "details": {
                    "used_vector_search": bool(vector_store.get("used_vector_search")),
                    "used_semantic_ranker": used_semantic_ranker,
                    "semantic_query_count": semantic_queries,
                    "semantic_query_cost_usd": semantic_unit_cost,
                    "semantic_cost_usd": round(semantic_cost, 6),
                    "results_returned": _safe_int(vector_store.get("results_returned")),
                    "vector_dimensions": _safe_int(vector_store.get("vector_dimensions")),
                    "retrieved_text_length": retrieved_text_length,
                    "result_chunk_count": len(result_chunks),
                    "request_duration_seconds": round(request_duration_seconds, 3),
                    "basic_tier_assumed": True,
                    "search_units": DEFAULT_AZURE_SEARCH_UNITS,
                    "basic_price_per_su_month_inr": AZURE_SEARCH_BASIC_PRICE_PER_SU_MONTH_INR,
                    "search_compute_cost_inr": round(azure_search_compute_cost_inr, 6),
                },
            },
            {
                "key": "speech",
                "label": "Speech",
                "cost_usd": round(speech_cost, 6),
                "cost_inr": round(speech_cost_inr, 6),
                "details": {
                    "tier": tier,
                    "stt_seconds": round(stt_seconds, 3),
                    "tts_characters": tts_characters,
                    "stt_cost_usd": round(stt_cost, 6),
                    "tts_cost_usd": round(tts_cost, 6),
                    "stt_cost_inr": round(stt_cost_inr, 6),
                    "tts_cost_inr": round(tts_cost_inr, 6),
                    "stt_cost_per_hour_usd": SPEECH_STT_COST_PER_HOUR_USD,
                    "tts_cost_per_million_chars_usd": SPEECH_TTS_COST_PER_MILLION_CHARS_USD,
                    "stt_realtime_price_per_hour_inr": SPEECH_STT_REALTIME_PRICE_PER_HOUR_INR,
                    "tts_neural_price_per_million_chars_inr": SPEECH_TTS_NEURAL_PRICE_PER_MILLION_CHARS_INR,
                    "payg_pricing_assumed": True,
                },
            },
            {
                "key": "cosmos",
                "label": "Cosmos DB",
                "cost_usd": round(cosmos_cost, 6),
                "details": {
                    "request_units": round(cosmos_request_units, 4),
                    "bytes_written": cosmos_bytes_written,
                    "read_operations": _safe_int(storage.get("reads")),
                    "write_operations": _safe_int(storage.get("writes")),
                    "pricing_note": "Current estimator exposes Cosmos telemetry and any mapped RU cost configuration.",
                },
            },
        ],
        "breakdown": [
            {
                "label": "LLM input tokens",
                "units": input_tokens,
                "rate": _safe_float(selected_pricing.get("input_per_1k_tokens_usd")),
                "cost_usd": round(llm_input_cost, 6),
            },
            {
                "label": "LLM output tokens",
                "units": output_tokens,
                "rate": _safe_float(selected_pricing.get("output_per_1k_tokens_usd")),
                "cost_usd": round(llm_output_cost, 6),
            },
            {
                "label": "Azure AI Search compute time",
                "units": round(request_duration_seconds, 3),
                "rate": round(
                    (
                        AZURE_SEARCH_BASIC_PRICE_PER_SU_MONTH_INR
                        * DEFAULT_AZURE_SEARCH_UNITS
                    )
                    / float(SECONDS_PER_30_DAY_MONTH),
                    9,
                ),
                "cost_usd": 0.0,
                "cost_inr": round(azure_search_compute_cost_inr, 6),
                "currency": "INR",
            },
            {
                "label": "Semantic ranker queries",
                "units": semantic_queries if used_semantic_ranker else 0,
                "rate": semantic_unit_cost,
                "cost_usd": round(semantic_cost, 6),
            },
            {
                "label": "Speech to text seconds",
                "units": round(stt_seconds, 3),
                "rate": SPEECH_STT_COST_PER_HOUR_USD,
                "cost_usd": round(stt_cost, 6),
                "cost_inr": round(stt_cost_inr, 6),
                "currency": "INR",
            },
            {
                "label": "Text to speech characters",
                "units": tts_characters,
                "rate": SPEECH_TTS_COST_PER_MILLION_CHARS_USD,
                "cost_usd": round(tts_cost, 6),
                "cost_inr": round(tts_cost_inr, 6),
                "currency": "INR",
            },
        ],
    }


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


@app.get("/cost-estimator")
def cost_estimator() -> str:
    return render_template("cost-estimator.html")


@app.get("/api/cost-estimator/data")
def get_cost_estimator_data() -> Any:
    return jsonify(
        {
            "ok": True,
            "models": _get_llm_pricing_options(),
            "requests": _list_request_metrics(),
            "default_model": runtime_config.model,
            "azure_tier": str(os.getenv("AZURE_TIER", "FREE")).strip().upper(),
            "semantic_query_cost_usd": _safe_float(
                os.getenv("AZURE_SEMANTIC_QUERY_COST", "0.001"), 0.001
            ),
        }
    )


@app.post("/api/cost-estimator/estimate")
def estimate_cost() -> Any:
    payload = request.get_json(silent=True) or {}
    metrics_name = str(payload.get("metrics_file") or "").strip()
    selected_model = str(payload.get("model") or "").strip()
    if not metrics_name:
        return jsonify({"ok": False, "error": "metrics_file is required"}), 400
    if not selected_model:
        return jsonify({"ok": False, "error": "model is required"}), 400

    try:
        metrics = _load_metrics_payload(metrics_name)
        metrics["_file_name"] = Path(metrics_name).name
        estimate = _estimate_request_cost(metrics, selected_model)
        return jsonify({"ok": True, "estimate": estimate})
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


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
    metrics = DocumentUploadMetricsRecorder(
        endpoint="/api/upload",
        file_name=uploaded.filename or "uploaded_file",
        index_name=index_name,
    )
    try:
        temp_path = _save_uploaded_file(uploaded)
        if not temp_path:
            return jsonify({"ok": False, "error": "Uploaded file is empty."}), 400
        metrics.update_file(
            file_path=temp_path,
            file_size_bytes=os.path.getsize(temp_path),
            index_name=index_name,
        )
        chunks, adi_pages = document_store.ingest_document(
            index_name=index_name,
            source_name=uploaded.filename or "uploaded_file",
            file_path=temp_path,
            chunk_size=runtime_config.chunk_size,
            overlap=runtime_config.chunk_overlap,
            vector_dimensions=runtime_config.embedding_dimensions,
            metrics_recorder=metrics,
        )
        if not chunks:
            metrics_path = metrics.finalize(
                status="error",
                error="No valid text chunks were produced. File content appears non-text or extraction failed.",
            )
            return jsonify(
                {
                    "ok": False,
                    "error": "No valid text chunks were produced. File content appears non-text or extraction failed.",
                    "metrics_file": _metrics_file_value(metrics_path),
                }
            ), 400
        embedding_bytes = len(chunks) * runtime_config.embedding_dimensions * 4
        cost_summary = evaluator.evaluate_ingestion(
            embedding_bytes_added=embedding_bytes,
            adi_pages_used=adi_pages,
        )
        metrics.record_cost_summary(
            cost_summary=cost_summary,
            upload_cost_usd=_estimate_adi_cost_usd(adi_pages),
        )
        metrics_path = metrics.finalize(status="success")
        return jsonify(
            {
                "ok": True,
                "index_name": index_name,
                "file_name": uploaded.filename,
                "chunks_ingested": len(chunks),
                "metrics_file": _metrics_file_value(metrics_path),
                "cost_summary": cost_summary,
                "store": document_store.stats(),
            }
        )
    except AzureQuotaExceededError as exc:
        metrics_path = metrics.finalize(status="quota_exceeded", error=str(exc))
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "metrics_file": _metrics_file_value(metrics_path),
                }
            ),
            429,
        )
    except Exception as exc:
        metrics_path = metrics.finalize(status="error", error=str(exc))
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "metrics_file": _metrics_file_value(metrics_path),
                }
            ),
            500,
        )
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
    metrics = RequestMetricsRecorder(
        endpoint="/api/chat",
        session_id=session_id,
        index_name=index_name,
        user_query=query,
    )

    try:
        cosmos_service = get_chat_history_service()
        if not session_id:
            session_id = cosmos_service.create_session(
                user_id="anonymous", metrics_recorder=metrics
            )
            logger.info(f"Created new session: {session_id}")
        metrics.update_request(session_id=session_id)

        cosmos_history = cosmos_service.get_session_history(
            session_id, metrics_recorder=metrics
        )

        contexts = document_store.search(
            query=query,
            index_name=index_name,
            top_k=top_k,
            use_semantic_ranker=use_semantic_ranker,
            use_vector_search=use_vector_search,
            vector_dimensions=runtime_config.embedding_dimensions,
            metrics_recorder=metrics,
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
            metrics_recorder=metrics,
        )
        cost_summary = evaluator.evaluate_query(
            input_tokens=int(llm_result["input_tokens"]),
            output_tokens=int(llm_result["output_tokens"]),
            use_semantic_ranker=use_semantic_ranker,
        )

        cosmos_service.save_message(
            session_id, "user", query, metrics_recorder=metrics
        )
        cosmos_service.save_message(
            session_id,
            "assistant",
            str(llm_result["answer"]),
            metrics_recorder=metrics,
        )

        chat_history.clear()
        chat_history.extend(cosmos_history[-20:])
        metrics.record_response(
            answer_characters=len(str(llm_result["answer"])),
            contexts_returned=len(context_payload),
        )
        metrics_path = metrics.finalize(status="success")

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
                "metrics_file": _metrics_file_value(metrics_path),
                "cost_summary": cost_summary,
            }
        )
    except AzureQuotaExceededError as exc:
        logger.error(f"Quota exceeded: {exc}")
        metrics_path = metrics.finalize(status="quota_exceeded", error=str(exc))
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "metrics_file": _metrics_file_value(metrics_path),
                }
            ),
            429,
        )
    except Exception as exc:
        logger.error(f"Chat error: {exc}", exc_info=True)
        metrics_path = metrics.finalize(status="error", error=str(exc))
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "metrics_file": _metrics_file_value(metrics_path),
                }
            ),
            500,
        )


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
    metrics = RequestMetricsRecorder(
        endpoint="/api/chat/stream",
        session_id=session_id,
        index_name=index_name,
        user_query=query,
    )

    try:
        cosmos_service = get_chat_history_service()
        if not session_id:
            session_id = cosmos_service.create_session(
                user_id="anonymous", metrics_recorder=metrics
            )
        metrics.update_request(session_id=session_id)

        cosmos_history = cosmos_service.get_session_history(
            session_id, metrics_recorder=metrics
        )

        contexts = document_store.search(
            query=query,
            index_name=index_name,
            top_k=top_k,
            use_semantic_ranker=use_semantic_ranker,
            use_vector_search=use_vector_search,
            vector_dimensions=runtime_config.embedding_dimensions,
            metrics_recorder=metrics,
        )
        context_texts = [c.text for c in contexts]
        context_payload = [
            {"chunk_id": c.chunk_id, "source_name": c.source_name, "text": c.text}
            for c in contexts
        ]
    except AzureQuotaExceededError as exc:
        logger.error(f"Stream chat setup quota exceeded: {exc}")
        metrics_path = metrics.finalize(status="quota_exceeded", error=str(exc))
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "metrics_file": _metrics_file_value(metrics_path),
                }
            ),
            429,
        )
    except Exception as exc:
        logger.error(f"Stream chat setup error: {exc}", exc_info=True)
        metrics_path = metrics.finalize(status="error", error=str(exc))
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "metrics_file": _metrics_file_value(metrics_path),
                }
            ),
            500,
        )

    def generate():
        try:
            stream_gen = llm_service.stream_rag_answer(
                query=query,
                contexts=context_texts,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=cosmos_history,
                metrics_recorder=metrics,
            )

            full_answer = ""
            token_count = 0
            meta = {}
            while True:
                try:
                    chunk = next(stream_gen)
                except StopIteration as stop:
                    if isinstance(stop.value, dict):
                        meta = stop.value
                    break
                full_answer += chunk
                token_count += 1
                yield f"data: {json.dumps({'type': 'token', 'token': chunk})}\n\n"

            cost_summary = evaluator.evaluate_query(
                input_tokens=meta.get(
                    "input_tokens",
                    max(int(len(" ".join(context_texts).split()) * 1.3), 1),
                ),
                output_tokens=meta.get("output_tokens", token_count),
                use_semantic_ranker=use_semantic_ranker,
            )

            cosmos_service.save_message(
                session_id, "user", query, metrics_recorder=metrics
            )
            cosmos_service.save_message(
                session_id, "assistant", full_answer, metrics_recorder=metrics
            )

            chat_history.clear()
            chat_history.extend(cosmos_history[-20:])
            metrics.record_response(
                answer_characters=len(full_answer),
                contexts_returned=len(context_payload),
            )
            metrics_path = metrics.finalize(status="success")

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
                "metrics_file": _metrics_file_value(metrics_path),
                "cost_summary": cost_summary,
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            logger.info(f"Stream chat: session={session_id}, tokens={token_count}")
        except Exception as exc:
            logger.error(f"Stream chat error: {exc}", exc_info=True)
            metrics_path = metrics.finalize(status="error", error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc), 'metrics_file': _metrics_file_value(metrics_path)})}\n\n"

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
    import logging

    logger = logging.getLogger(__name__)

    if "file" not in request.files:
        return jsonify({"ok": False, "error": "audio file is required"}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"ok": False, "error": "filename is required"}), 400

    temp_audio_path = ""
    metrics = RequestMetricsRecorder(endpoint="/api/voice-chat")
    try:
        suffix = Path(uploaded_file.filename or "").suffix or ".wav"
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, "voice" + suffix)
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.read())
        logger.info(
            f"Voice chat: filename={uploaded_file.filename}, suffix={suffix}, temp_path={temp_audio_path}, size={os.path.getsize(temp_audio_path)}"
        )
        with open(temp_audio_path, "rb") as f:
            header = f.read(32)
            logger.info(f"Voice chat: file header hex: {header.hex()}")
            logger.info(f"Voice chat: file header ascii: {repr(header[:16])}")

        if not temp_audio_path or not os.path.exists(temp_audio_path):
            return jsonify({"ok": False, "error": "Failed to save audio file."}), 500

        speech_service = get_speech_service()
        query = speech_service.transcribe_audio(
            temp_audio_path, metrics_recorder=metrics
        )
        metrics.update_request(user_query=query)

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
        metrics.update_request(session_id=session_id, index_name=index_name)

        cosmos_service = get_chat_history_service()
        if not session_id:
            session_id = cosmos_service.create_session(
                user_id="anonymous", metrics_recorder=metrics
            )
        metrics.update_request(session_id=session_id)

        cosmos_history = cosmos_service.get_session_history(
            session_id, metrics_recorder=metrics
        )

        contexts = document_store.search(
            query=query,
            index_name=index_name,
            top_k=top_k,
            use_semantic_ranker=use_semantic_ranker,
            use_vector_search=use_vector_search,
            vector_dimensions=runtime_config.embedding_dimensions,
            metrics_recorder=metrics,
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
            metrics_recorder=metrics,
        )
        cost_summary = evaluator.evaluate_query(
            input_tokens=int(llm_result["input_tokens"]),
            output_tokens=int(llm_result["output_tokens"]),
            use_semantic_ranker=use_semantic_ranker,
        )

        cosmos_service.save_message(
            session_id, "user", query, metrics_recorder=metrics
        )
        cosmos_service.save_message(
            session_id,
            "assistant",
            str(llm_result["answer"]),
            metrics_recorder=metrics,
        )

        chat_history.clear()
        chat_history.extend(cosmos_history[-20:])

        answer_text = str(llm_result["answer"])
        tts_audio_data = speech_service.synthesize_speech(
            answer_text, metrics_recorder=metrics
        )
        tts_audio_base64 = base64.b64encode(tts_audio_data).decode("utf-8")
        metrics.record_response(
            answer_characters=len(answer_text),
            contexts_returned=len(context_payload),
        )
        metrics_path = metrics.finalize(status="success")

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
                "metrics_file": _metrics_file_value(metrics_path),
                "cost_summary": cost_summary,
            }
        )
    except AzureQuotaExceededError as exc:
        metrics_path = metrics.finalize(status="quota_exceeded", error=str(exc))
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "metrics_file": _metrics_file_value(metrics_path),
                }
            ),
            429,
        )
    except Exception as exc:
        metrics_path = metrics.finalize(status="error", error=str(exc))
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "metrics_file": _metrics_file_value(metrics_path),
                }
            ),
            500,
        )
    finally:
        if temp_audio_path:
            temp_dir = os.path.dirname(temp_audio_path)
            if os.path.exists(temp_dir):
                try:
                    import shutil

                    shutil.rmtree(temp_dir, ignore_errors=True)
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
    metrics = RequestMetricsRecorder(
        endpoint="/api/chat-with-tts",
        session_id=session_id,
        index_name=index_name,
        user_query=query,
    )

    try:
        cosmos_service = get_chat_history_service()
        if not session_id:
            session_id = cosmos_service.create_session(
                user_id="anonymous", metrics_recorder=metrics
            )
        metrics.update_request(session_id=session_id)

        cosmos_history = cosmos_service.get_session_history(
            session_id, metrics_recorder=metrics
        )

        contexts = document_store.search(
            query=query,
            index_name=index_name,
            top_k=top_k,
            use_semantic_ranker=use_semantic_ranker,
            use_vector_search=use_vector_search,
            vector_dimensions=runtime_config.embedding_dimensions,
            metrics_recorder=metrics,
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
            metrics_recorder=metrics,
        )
        cost_summary = evaluator.evaluate_query(
            input_tokens=int(llm_result["input_tokens"]),
            output_tokens=int(llm_result["output_tokens"]),
            use_semantic_ranker=use_semantic_ranker,
        )

        cosmos_service.save_message(
            session_id, "user", query, metrics_recorder=metrics
        )
        cosmos_service.save_message(
            session_id,
            "assistant",
            str(llm_result["answer"]),
            metrics_recorder=metrics,
        )

        chat_history.clear()
        chat_history.extend(cosmos_history[-20:])

        answer_text = str(llm_result["answer"])

        speech_service = get_speech_service()
        tts_audio_data = speech_service.synthesize_speech(
            answer_text, metrics_recorder=metrics
        )
        tts_audio_base64 = base64.b64encode(tts_audio_data).decode("utf-8")
        metrics.record_response(
            answer_characters=len(answer_text),
            contexts_returned=len(context_payload),
        )
        metrics_path = metrics.finalize(status="success")

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
                "metrics_file": _metrics_file_value(metrics_path),
                "cost_summary": cost_summary,
            }
        )
    except AzureQuotaExceededError as exc:
        metrics_path = metrics.finalize(status="quota_exceeded", error=str(exc))
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "metrics_file": _metrics_file_value(metrics_path),
                }
            ),
            429,
        )
    except Exception as exc:
        metrics_path = metrics.finalize(status="error", error=str(exc))
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "metrics_file": _metrics_file_value(metrics_path),
                }
            ),
            500,
        )


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
    metrics = RequestMetricsRecorder(
        endpoint="/api/tts-generate",
        user_query=(str(text).strip()[:120] or "tts-generate"),
    )
    try:
        speech_service = get_speech_service()
        audio_data = speech_service.synthesize_speech(
            text, metrics_recorder=metrics
        )
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        metrics_path = metrics.finalize(status="success")
        return jsonify(
            {
                "ok": True,
                "audio": audio_base64,
                "metrics_file": _metrics_file_value(metrics_path),
            }
        )
    except AzureQuotaExceededError as exc:
        metrics_path = metrics.finalize(status="quota_exceeded", error=str(exc))
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "metrics_file": _metrics_file_value(metrics_path),
                }
            ),
            429,
        )
    except Exception as exc:
        metrics_path = metrics.finalize(status="error", error=str(exc))
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "metrics_file": _metrics_file_value(metrics_path),
                }
            ),
            500,
        )


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
