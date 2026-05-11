"""Aggregate router. Versioned under /v1."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import (
    agents,
    chat,
    conversations,
    files,
    health,
    memory,
    models,
    personalities,
    voice,
)

api_router = APIRouter()
api_router.include_router(health.router)

v1 = APIRouter(prefix="/v1")
v1.include_router(chat.router)
v1.include_router(conversations.router)
v1.include_router(models.router)
v1.include_router(memory.router)
v1.include_router(voice.router)
v1.include_router(files.router)
v1.include_router(personalities.router)
v1.include_router(agents.router)

api_router.include_router(v1)
