"""Health + capability discovery. Used by the mobile app to bootstrap."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.config import settings
from app.llm.registry import list_providers
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    providers = {p["id"]: p["available"] for p in list_providers()}
    return HealthResponse(version=__version__, env=settings.app_env, providers=providers)
