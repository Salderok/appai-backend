"""Ollama provider — local LLMs through the Ollama HTTP API (optional, offline fallback)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from app.config import settings
from app.core.exceptions import ProviderError
from app.llm.base import BaseLLMProvider
from app.llm.types import ChatChunk, ChatCompletion, ChatMessage, ChatOptions


class OllamaProvider(BaseLLMProvider):
    id = "ollama"
    display_name = "Ollama (local)"

    DEFAULT_MODELS = ["phi3:mini", "gemma2:2b", "qwen2.5:3b"]

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (base_url or settings.ollama_base_url or "").rstrip("/")

    def is_available(self) -> bool:
        return bool(self._base_url)

    def list_models(self) -> list[str]:
        return list(self.DEFAULT_MODELS)

    @staticmethod
    def _to_ollama(messages: list[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def complete(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> ChatCompletion:
        if not self._base_url:
            raise ProviderError("Ollama base URL not configured.")
        payload = {
            "model": options.model,
            "messages": self._to_ollama(messages),
            "options": {
                k: v
                for k, v in {
                    "temperature": options.temperature,
                    "num_predict": options.max_tokens,
                    "top_p": options.top_p,
                    "stop": options.stop,
                }.items()
                if v is not None
            },
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(f"{self._base_url}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Ollama error: {exc}") from exc

        msg = data.get("message", {}).get("content", "")
        return ChatCompletion(
            content=msg,
            finish_reason="stop",
            model=options.model,
            provider=self.id,
            usage={"eval_count": data.get("eval_count")},
            raw=data,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[ChatChunk]:
        if not self._base_url:
            raise ProviderError("Ollama base URL not configured.")
        payload = {
            "model": options.model,
            "messages": self._to_ollama(messages),
            "options": {
                k: v
                for k, v in {
                    "temperature": options.temperature,
                    "num_predict": options.max_tokens,
                    "top_p": options.top_p,
                    "stop": options.stop,
                }.items()
                if v is not None
            },
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST", f"{self._base_url}/api/chat", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        event = json.loads(line)
                        delta = event.get("message", {}).get("content", "")
                        if delta:
                            yield ChatChunk(delta=delta)
                        if event.get("done"):
                            yield ChatChunk(
                                delta="",
                                finish_reason="stop",
                                usage={"eval_count": event.get("eval_count")},
                            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Ollama error: {exc}") from exc
