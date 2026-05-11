"""Agent request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    task: str = Field(min_length=1, description="What the agent should do.")
    provider: str | None = None
    model: str | None = None
    conversation_id: str | None = None
    max_steps: int = Field(default=6, ge=1, le=12)
    system_prompt: str | None = None
