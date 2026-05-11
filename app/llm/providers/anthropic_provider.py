"""Anthropic Claude provider."""

from __future__ import annotations

from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from app.config import settings
from app.core.exceptions import ProviderError
from app.llm.base import BaseLLMProvider
from app.llm.types import ChatChunk, ChatCompletion, ChatMessage, ChatOptions


class AnthropicProvider(BaseLLMProvider):
    id = "anthropic"
    display_name = "Anthropic"

    DEFAULT_MODELS = [
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
        "claude-3-opus-latest",
    ]

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.anthropic_api_key
        self._client: AsyncAnthropic | None = None

    def is_available(self) -> bool:
        return bool(self._api_key)

    def list_models(self) -> list[str]:
        return list(self.DEFAULT_MODELS)

    @property
    def client(self) -> AsyncAnthropic:
        if self._client is None:
            if not self._api_key:
                raise ProviderError("Anthropic API key not configured.")
            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    @staticmethod
    def _split_system(messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
        """Anthropic puts the system prompt in a top-level field, not in messages."""
        system_parts: list[str] = []
        rest: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            role = "assistant" if m.role == "assistant" else "user"
            rest.append({"role": role, "content": m.content})
        return ("\n\n".join(system_parts) if system_parts else None), rest

    async def complete(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> ChatCompletion:
        system, msgs = self._split_system(messages)
        try:
            resp = await self.client.messages.create(
                model=options.model,
                system=system or "",
                messages=msgs,
                max_tokens=options.max_tokens or 1024,
                temperature=options.temperature,
                top_p=options.top_p,
                stop_sequences=options.stop,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Anthropic error: {exc}") from exc

        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        return ChatCompletion(
            content=text,
            finish_reason=resp.stop_reason,
            model=resp.model,
            provider=self.id,
            usage=usage,
            raw=resp,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[ChatChunk]:
        system, msgs = self._split_system(messages)
        try:
            async with self.client.messages.stream(
                model=options.model,
                system=system or "",
                messages=msgs,
                max_tokens=options.max_tokens or 1024,
                temperature=options.temperature,
                top_p=options.top_p,
                stop_sequences=options.stop,
            ) as stream:
                async for text in stream.text_stream:
                    yield ChatChunk(delta=text)
                final = await stream.get_final_message()
                yield ChatChunk(
                    delta="",
                    finish_reason=final.stop_reason,
                    usage={
                        "input_tokens": final.usage.input_tokens,
                        "output_tokens": final.usage.output_tokens,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Anthropic error: {exc}") from exc
