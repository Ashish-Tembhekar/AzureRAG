import os
from typing import Dict, Generator, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .request_metrics import RequestMetricsRecorder

load_dotenv()


def _estimate_tokens(text: str) -> int:
    # Conservative fallback when provider usage fields are absent.
    return max(int(len(text.split()) * 1.3), 1)


async def normalize_text_for_tts(text: str) -> str:
    """
    Reference-compatible async helper that normalizes text for TTS.
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
    )
    model = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "Normalize text for TTS: preserve meaning, remove markdown artifacts.",
            },
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content or text


class NvidiaLLMService:
    def __init__(self) -> None:
        self.base_url = "https://integrate.api.nvidia.com/v1"

    def _client(self) -> OpenAI:
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set. Add it to your .env file.")
        return OpenAI(base_url=self.base_url, api_key=api_key)

    def generate_rag_answer(
        self,
        query: str,
        contexts: List[str],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 600,
        chat_history: Optional[List[Dict[str, str]]] = None,
        metrics_recorder: Optional[RequestMetricsRecorder] = None,
    ) -> Dict[str, object]:
        selected_model = model or os.getenv(
            "NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"
        )
        context_block = "\n\n".join(
            f"[Context {i + 1}] {c}" for i, c in enumerate(contexts)
        )
        system_prompt = (
            "You are a precise RAG assistant. Use only the supplied context for factual claims. "
            "If context is insufficient, say so explicitly and suggest what is missing."
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for item in chat_history[-8:]:
                role = item.get("role", "")
                content = item.get("content", "")
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})

        messages.append(
            {
                "role": "user",
                "content": f"Context:\n{context_block or 'No context available.'}\n\nQuestion: {query}",
            }
        )

        client = self._client()
        response = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        if prompt_tokens is None:
            prompt_tokens = _estimate_tokens(" ".join(m["content"] for m in messages))
        if completion_tokens is None:
            completion_tokens = _estimate_tokens(answer)

        result = {
            "answer": answer,
            "input_tokens": int(prompt_tokens),
            "output_tokens": int(completion_tokens),
            "model": selected_model,
        }
        if metrics_recorder is not None:
            metrics_recorder.record_llm(
                model=selected_model,
                input_tokens=int(prompt_tokens),
                output_tokens=int(completion_tokens),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return result

    def stream_rag_answer(
        self,
        query: str,
        contexts: List[str],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 600,
        chat_history: Optional[List[Dict[str, str]]] = None,
        metrics_recorder: Optional[RequestMetricsRecorder] = None,
    ) -> Generator[str, None, Dict[str, object]]:
        selected_model = model or os.getenv(
            "NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"
        )
        context_block = "\n\n".join(
            f"[Context {i + 1}] {c}" for i, c in enumerate(contexts)
        )
        system_prompt = (
            "You are a precise RAG assistant. Use only the supplied context for factual claims. "
            "If context is insufficient, say so explicitly and suggest what is missing."
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for item in chat_history[-8:]:
                role = item.get("role", "")
                content = item.get("content", "")
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})

        messages.append(
            {
                "role": "user",
                "content": f"Context:\n{context_block or 'No context available.'}\n\nQuestion: {query}",
            }
        )

        client = self._client()
        response = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        full_answer = ""
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                full_answer += delta.content
                yield delta.content

        input_tokens = _estimate_tokens(" ".join(m["content"] for m in messages))
        output_tokens = _estimate_tokens(full_answer)

        result = {
            "answer": full_answer,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "model": selected_model,
        }
        if metrics_recorder is not None:
            metrics_recorder.record_llm(
                model=selected_model,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return result


_llm_singleton: Optional[NvidiaLLMService] = None


def get_llm_service() -> NvidiaLLMService:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = NvidiaLLMService()
    return _llm_singleton
