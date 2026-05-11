"""The owner of the assistant. We keep a row even in single-user mode."""

from __future__ import annotations

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    display_name: Mapped[str] = mapped_column(String(120), default="Owner")
    # Reserved for future multi-user support
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
