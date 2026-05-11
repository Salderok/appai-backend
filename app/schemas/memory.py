"""Memory schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MemoryItemOut(BaseModel):
    id: str
    kind: str
    content: str
    importance: float
    source_conversation_id: str | None = None
    created_at: str
    updated_at: str


class MemoryCreate(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    kind: str = "fact"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class MemorySearchResult(BaseModel):
    item: MemoryItemOut
    score: float
