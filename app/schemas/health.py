"""Health check schemas."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    env: str
    providers: dict[str, bool]
