"""Pydantic schemas for /chat endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ChatMessageIn(BaseModel):
    role: Role
    content: str
    name: str | None = None


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    provider: str | None = Field(
        default=None,
        description="Provider id: openai | anthropic | gemini | deepseek | ollama. "
        "Falls back to the conversation's stored provider, then the default.",
    )
    model: str | None = None
    messages: list[ChatMessageIn] = Field(default_factory=list)
    system_prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=32_000)
    stream: bool = True
    use_memory: bool = True
    personality_id: str | None = None
    attachments: list[str] | None = None  # uploaded file ids


class ChatMessageOut(BaseModel):
    id: str
    role: Role
    content: str
    metadata: dict[str, Any] | None = None
    created_at: str


class ChatResponse(BaseModel):
    conversation_id: str
    message: ChatMessageOut
    usage: dict[str, Any] | None = None
