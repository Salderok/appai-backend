"""Conversation list/detail schemas."""

from __future__ import annotations

from pydantic import BaseModel


class ConversationSummary(BaseModel):
    id: str
    title: str
    provider: str
    model: str
    created_at: str
    updated_at: str
    message_count: int


class ConversationCreate(BaseModel):
    title: str | None = None
    system_prompt: str | None = None
    personality_id: str | None = None
    provider: str | None = None
    model: str | None = None


class ConversationUpdate(BaseModel):
    title: str | None = None
    system_prompt: str | None = None
    personality_id: str | None = None
    provider: str | None = None
    model: str | None = None


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: str
    metadata: dict | None = None


class ConversationDetail(ConversationSummary):
    system_prompt: str | None = None
    messages: list[MessageOut]
