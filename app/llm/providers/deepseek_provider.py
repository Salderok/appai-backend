"""DeepSeek provider. DeepSeek exposes an OpenAI-compatible API, so we reuse the OpenAI client."""

from __future__ import annotations

from app.config import settings
from app.llm.providers.openai_provider import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    DEFAULT_MODELS = ["deepseek-chat", "deepseek-reasoner"]

    def __init__(self) -> None:
        super().__init__(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            provider_id="deepseek",
            display_name="DeepSeek",
            models=self.DEFAULT_MODELS,
        )
