"""A single message in a conversation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.conversation import Conversation


class Message(Base):
    __tablename__ = "messages"

    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))  # "user" | "assistant" | "system" | "tool"
    content: Mapped[str] = mapped_column(Text)
    # Provider response metadata: model, finish_reason, tokens, tool_calls, attachments.
    extra: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)

    conversation: Mapped[Conversation] = relationship("Conversation", back_populates="messages")
