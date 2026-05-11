"""Provider registry — single source of truth for getting an LLM provider by id.

Routes and services must call `get_provider(...)` here rather than instantiating
providers directly. Adding a new provider = adding it to `_build_registry`.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.core.exceptions import ConfigError, ProviderError
from app.llm.base import BaseLLMProvider
from app.llm.providers.anthropic_provider import AnthropicProvider
from app.llm.providers.deepseek_provider import DeepSeekProvider
from app.llm.providers.gemini_provider import GeminiProvider
from app.llm.providers.ollama_provider import OllamaProvider
from app.llm.providers.openai_provider import OpenAIProvider


@lru_cache(maxsize=1)
def _build_registry() -> dict[str, BaseLLMProvider]:
    return {
        "openai": OpenAIProvider(),
        "anthropic": AnthropicProvider(),
        "gemini": GeminiProvider(),
        "deepseek": DeepSeekProvider(),
        "ollama": OllamaProvider(),
    }


def list_providers() -> list[dict]:
    """Provider catalog for the mobile settings screen."""
    return [
        {
            "id": p.id,
            "display_name": p.display_name,
            "available": p.is_available(),
            "models": p.list_models(),
        }
        for p in _build_registry().values()
    ]


def get_provider(provider_id: str | None = None) -> BaseLLMProvider:
    pid = (provider_id or settings.default_provider).lower()
    registry = _build_registry()
    if pid not in registry:
        raise ConfigError(f"Unknown LLM provider: {pid}")
    provider = registry[pid]
    if not provider.is_available():
        raise ProviderError(
            f"Provider {provider.display_name} is not configured (missing API key or base URL)."
        )
    return provider


def get_embedding_provider() -> BaseLLMProvider:
    """For now embeddings always go through OpenAI (cheap and fast)."""
    if settings.embedding_provider != "openai":
        raise ConfigError(
            "Only OpenAI embeddings are wired up. Set EMBEDDING_PROVIDER=openai or implement a local one."
        )
    return get_provider("openai")
