"""Memory CRUD + search routes."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.db.models.memory import MemoryItem
from app.deps import DbSession, DeviceKey, MemoryDep
from app.schemas.memory import MemoryCreate, MemoryItemOut, MemorySearchResult

router = APIRouter(prefix="/memory", tags=["memory"])


def _to_out(m: MemoryItem) -> MemoryItemOut:
    return MemoryItemOut(
        id=m.id,
        kind=m.kind,
        content=m.content,
        importance=m.importance,
        source_conversation_id=m.source_conversation_id,
        created_at=m.created_at.isoformat(),
        updated_at=m.updated_at.isoformat(),
    )


@router.get("", response_model=list[MemoryItemOut])
async def list_memory(memory: MemoryDep, _: DeviceKey, limit: int = 100) -> list[MemoryItemOut]:
    items = await memory.vector.list_all(limit=limit)
    return [_to_out(m) for m in items]


@router.post("", response_model=MemoryItemOut)
async def create_memory(
    payload: MemoryCreate, memory: MemoryDep, _: DeviceKey
) -> MemoryItemOut:
    item = await memory.vector.add(
        content=payload.content, kind=payload.kind, importance=payload.importance
    )
    return _to_out(item)


@router.get("/search", response_model=list[MemorySearchResult])
async def search_memory(
    memory: MemoryDep,
    _: DeviceKey,
    q: str = Query(min_length=1),
    k: int = 5,
) -> list[MemorySearchResult]:
    results = await memory.vector.search(q, k=k)
    return [MemorySearchResult(item=_to_out(item), score=score) for item, score in results]


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(memory_id: str, memory: MemoryDep, _: DbSession, __: DeviceKey) -> None:
    await memory.vector.delete(memory_id)
