"""Long-term semantic memory.

Storage:
  - `MemoryItem.embedding` is stored as a JSON list of floats — portable across
    SQLite (dev) and Postgres (prod). On Postgres + pgvector you can switch this
    column to `Vector(N)` in an Alembic migration and replace `_top_k_python`
    below with a server-side `<=>` query. The interface stays the same.

Embeddings always go through the OpenAI embedding model (configurable).
"""

from __future__ import annotations

import math

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.memory import MemoryItem
from app.llm.registry import get_embedding_provider


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=False))
    da = math.sqrt(sum(x * x for x in a))
    db_ = math.sqrt(sum(y * y for y in b))
    return num / (da * db_) if da and db_ else 0.0


class VectorMemory:
    """Thin façade over `MemoryItem` with embedding-aware add/search."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def add(
        self,
        *,
        content: str,
        kind: str = "fact",
        importance: float = 0.5,
        source_conversation_id: str | None = None,
        metadata: dict | None = None,
    ) -> MemoryItem:
        embedding = await self._embed_one(content)
        item = MemoryItem(
            content=content,
            kind=kind,
            importance=importance,
            embedding=embedding,
            source_conversation_id=source_conversation_id,
            extra=metadata,
        )
        self.db.add(item)
        await self.db.flush()
        return item

    async def search(self, query: str, *, k: int = 5, min_score: float = 0.2) -> list[tuple[MemoryItem, float]]:
        if not query.strip():
            return []
        q = await self._embed_one(query)
        rows = list((await self.db.execute(select(MemoryItem))).scalars().all())
        scored: list[tuple[MemoryItem, float]] = []
        for m in rows:
            if not m.embedding:
                continue
            score = cosine(q, m.embedding) + 0.05 * (m.importance or 0.0)
            if score >= min_score:
                scored.append((m, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    async def list_all(self, *, limit: int = 100) -> list[MemoryItem]:
        stmt = select(MemoryItem).order_by(MemoryItem.updated_at.desc()).limit(limit)
        return list((await self.db.execute(stmt)).scalars().all())

    async def delete(self, memory_id: str) -> None:
        item = await self.db.get(MemoryItem, memory_id)
        if item:
            await self.db.delete(item)

    async def _embed_one(self, text: str) -> list[float]:
        provider = get_embedding_provider()
        [vec] = await provider.embed([text])
        return vec
