"""AI personality preset: system prompt + tone parameters."""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Personality(Base):
    __tablename__ = "personalities"

    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    system_prompt: Mapped[str] = mapped_column(Text)
    # voice, formality, temperature, etc.
    params: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)
    is_builtin: Mapped[bool] = mapped_column(default=False)
