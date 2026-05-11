"""OpenAI ChatCompletions provider. Also used (with a different base_url) for DeepSeek."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from app.config import settings
from app.core.exceptions import ProviderError
from app.llm.base import BaseLLMProvider
from app.llm.types import ChatChunk, ChatCompletion, ChatMessage, ChatOptions


class OpenAIProvider(BaseLLMProvider):
    id = "openai"
    display_name = "OpenAI"

    DEFAULT_MODELS = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "o4-mini",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        provider_id: str = "openai",
        display_name: str = "OpenAI",
        models: list[str] | None = None,
    ) -> None:
        self.id = provider_id
        self.display_name = display_name
        self._api_key = api_key or settings.openai_api_key
        self._base_url = base_url or settings.openai_base_url
        self._models = models or self.DEFAULT_MODELS
        self._client: AsyncOpenAI | None = None

    def is_available(self) -> bool:
        return bool(self._api_key)

    def list_models(self) -> list[str]:
        return list(self._models)

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            if not self._api_key:
                raise ProviderError(f"{self.display_name} API key not configured.")
            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    @staticmethod
    def _to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            entry: dict[str, Any] = {"role": m.role}
            if m.images and m.role == "user":
                parts: list[dict[str, Any]] = [{"type": "text", "text": m.content}]
                for img in m.images:
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": img.data_uri, "detail": img.detail},
                    })
                entry["content"] = parts
            else:
                entry["content"] = m.content
            if m.name:
                entry["name"] = m.name
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            # Pass tool_calls back to the model (assistant turns from prior steps).
            if m.role == "assistant":
                tool_calls = (m.metadata or {}).get("tool_calls")
                if tool_calls:
                    entry["tool_calls"] = tool_calls
            out.append(entry)
        return out

    async def complete(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> ChatCompletion:
        kwargs: dict[str, Any] = {
            "model": options.model,
            "messages": self._to_openai_messages(messages),
            "temperature": options.temperature,
            "max_tokens": options.max_tokens,
            "top_p": options.top_p,
            "stop": options.stop,
            "stream": False,
        }
        if options.tools:
            kwargs["tools"] = options.tools
        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"{self.display_name} error: {exc}") from exc

        choice = resp.choices[0]
        return ChatCompletion(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            model=resp.model,
            provider=self.id,
            usage=resp.usage.model_dump() if resp.usage else None,
            raw=resp,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[ChatChunk]:
        try:
            stream = await self.client.chat.completions.create(
                model=options.model,
                messages=self._to_openai_messages(messages),
                temperature=options.temperature,
                max_tokens=options.max_tokens,
                top_p=options.top_p,
                stop=options.stop,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"{self.display_name} error: {exc}") from exc

        async for event in stream:
            if not event.choices:
                if event.usage:
                    yield ChatChunk(delta="", usage=event.usage.model_dump())
                continue
            choice = event.choices[0]
            delta = (choice.delta.content or "") if choice.delta else ""
            yield ChatChunk(
                delta=delta,
                finish_reason=choice.finish_reason,
                raw=event,
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = await self.client.embeddings.create(
                model=settings.embedding_model,
                input=texts,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"{self.display_name} embedding error: {exc}") from exc
        return [item.embedding for item in resp.data]
