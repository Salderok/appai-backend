"""Common FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import require_device_key
from app.db.session import get_session
from app.memory.manager import MemoryManager

DbSession = Annotated[AsyncSession, Depends(get_session)]
DeviceKey = Annotated[str, Depends(require_device_key)]


async def get_memory_manager(db: DbSession) -> MemoryManager:
    return MemoryManager(db)


MemoryDep = Annotated[MemoryManager, Depends(get_memory_manager)]
