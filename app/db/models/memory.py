"""Long-term memory items with optional embeddings.

We store the embedding column conditionally so the codebase also works on
SQLite (dev). On Postgres + pgvector we'd switch this to `Vector(1536)`
through an Alembic migration; for now we use JSON to stay portable.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class MemoryItem(Base):
    __tablename__ = "memory_items"

    user_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    kind: Mapped[str] = mapped_column(String(40), default="fact")
    # "fact" | "preference" | "summary" | "task" | "skill"
    content: Mapped[str] = mapped_column(Text)
    source_conversation_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    importance: Mapped[float] = mapped_column(default=0.5)
    # Stored as JSON list of floats for portability. Real production:
    # use sqlalchemy_pgvector.Vector and run cosine search server-side.
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    extra: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)
