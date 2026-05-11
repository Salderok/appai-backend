"""LLM-driven conversation titling.

Runs once per conversation, after the first user/assistant turn, to generate
a 3-6 word title. Falls back silently to the first message line if the LLM
call fails — we never want titling to break a chat.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.llm.base import BaseLLMProvider
from app.llm.types import ChatMessage, ChatOptions

logger = get_logger(__name__)


TITLE_SYSTEM = (
    "You write extremely short conversation titles. "
    "Respond with 3-6 words, no quotes, no trailing punctuation, no emoji."
)


async def generate_title(
    provider: BaseLLMProvider,
    *,
    user_message: str,
    assistant_message: str,
    model: str,
) -> str | None:
    prompt = (
        f"User: {user_message[:400]}\n\n"
        f"Assistant: {assistant_message[:400]}\n\n"
        "Title:"
    )
    try:
        completion = await provider.complete(
            [
                ChatMessage(role="system", content=TITLE_SYSTEM),
                ChatMessage(role="user", content=prompt),
            ],
            ChatOptions(model=model, temperature=0.3, max_tokens=20),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("title_generation_failed", error=str(exc))
        return None

    title = completion.content.strip().strip('"').strip("'").rstrip(".!?")
    return title[:80] if title else None
