"""LLM provider interface. Every provider implements the same shape."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.llm.types import ChatChunk, ChatCompletion, ChatMessage, ChatOptions


class BaseLLMProvider(ABC):
    """Common shape every provider implements.

    Implementations must be async and avoid blocking IO. Providers should
    raise `app.core.exceptions.ProviderError` for retryable upstream issues.
    """

    id: str  # short identifier, e.g. "openai"
    display_name: str  # human-facing, e.g. "OpenAI"

    @abstractmethod
    def is_available(self) -> bool:
        """True iff this provider is fully configured (API key present, etc.)."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return curated default model ids the UI should expose."""

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> ChatCompletion: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[ChatChunk]: ...

    async def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError(f"{self.id} does not implement embeddings.")
