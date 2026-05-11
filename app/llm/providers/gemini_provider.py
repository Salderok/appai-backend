"""Google Gemini provider via the current `google-genai` SDK."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from app.config import settings
from app.core.exceptions import ProviderError
from app.llm.base import BaseLLMProvider
from app.llm.types import ChatChunk, ChatCompletion, ChatMessage, ChatOptions

if TYPE_CHECKING:
    from google.genai import Client as GenAIClient


class GeminiProvider(BaseLLMProvider):
    id = "gemini"
    display_name = "Google Gemini"

    DEFAULT_MODELS = [
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.gemini_api_key
        self._client: GenAIClient | None = None

    def is_available(self) -> bool:
        return bool(self._api_key)

    def list_models(self) -> list[str]:
        return list(self.DEFAULT_MODELS)

    @property
    def client(self) -> GenAIClient:
        if self._client is None:
            if not self._api_key:
                raise ProviderError("Gemini API key not configured.")
            from google import genai  # local import: avoid paying cost unless used
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @staticmethod
    def _to_gemini(messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
        """Split into system instruction + contents (role+parts)."""
        system_parts: list[str] = []
        contents: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.content}]})
        return ("\n\n".join(system_parts) if system_parts else None), contents

    def _generation_config(self, options: ChatOptions, system: str | None) -> dict:
        from google.genai import types as genai_types

        cfg = {
            "temperature": options.temperature,
            "max_output_tokens": options.max_tokens,
            "top_p": options.top_p,
            "stop_sequences": options.stop,
            "system_instruction": system,
        }
        return genai_types.GenerateContentConfig(**{k: v for k, v in cfg.items() if v is not None})

    async def complete(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> ChatCompletion:
        system, contents = self._to_gemini(messages)
        try:
            resp = await self.client.aio.models.generate_content(
                model=options.model,
                contents=contents,
                config=self._generation_config(options, system),
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Gemini error: {exc}") from exc

        text = getattr(resp, "text", "") or ""
        finish = None
        if getattr(resp, "candidates", None):
            finish = getattr(resp.candidates[0], "finish_reason", None)
        return ChatCompletion(
            content=text,
            finish_reason=str(finish) if finish else None,
            model=options.model,
            provider=self.id,
            usage=None,
            raw=resp,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[ChatChunk]:
        system, contents = self._to_gemini(messages)
        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=options.model,
                contents=contents,
                config=self._generation_config(options, system),
            )
            async for event in stream:
                text = getattr(event, "text", "") or ""
                if text:
                    yield ChatChunk(delta=text)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Gemini error: {exc}") from exc
