"""A chat conversation thread."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.message import Message


class Conversation(Base):
    __tablename__ = "conversations"

    user_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(200), default="New chat")
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    personality_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("personalities.id"), nullable=True
    )
    provider: Mapped[str] = mapped_column(String(40), default="openai")
    model: Mapped[str] = mapped_column(String(80), default="gpt-4o-mini")

    messages: Mapped[list[Message]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
