"""List configured providers and their default models."""

from __future__ import annotations

from fastapi import APIRouter

from app.deps import DeviceKey
from app.llm.registry import list_providers

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
async def get_models(_: DeviceKey) -> dict:
    return {"providers": list_providers()}
