"""Memory manager facade.

Combines:
  - short_term: token-budgeted recent buffer (handled inline in chat_service)
  - long_term:  semantic retrieval over MemoryItem rows
  - episodic:   LLM-driven fact extraction
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging import get_logger
from app.llm.registry import get_provider
from app.memory.episodic import FactExtractor
from app.memory.long_term import VectorMemory

logger = get_logger(__name__)


class MemoryManager:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.vector = VectorMemory(db)

    async def build_context(self, query: str) -> str | None:
        """Return a system-prompt fragment containing relevant memories, or None."""
        try:
            results = await self.vector.search(query, k=5)
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory_search_failed", error=str(exc))
            return None
        if not results:
            return None

        lines = ["Relevant long-term memory about the user (use only if relevant):"]
        for item, score in results:
            tag = item.kind
            lines.append(f"- [{tag} · {score:.2f}] {item.content}")
        return "\n".join(lines)

    async def observe_turn(
        self,
        *,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Extract durable facts from a finished turn and persist them."""
        if not user_message or not assistant_message:
            return
        try:
            provider = get_provider(settings.default_provider)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fact_provider_unavailable", error=str(exc))
            return

        extractor = FactExtractor(provider, self.vector, model=settings.default_model)
        await extractor.extract_and_store(
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )
