"""Personality preset CRUD."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.db.models.personality import Personality
from app.deps import DbSession, DeviceKey
from app.schemas.personality import PersonalityCreate, PersonalityOut, PersonalityUpdate
from app.services import personality_service

router = APIRouter(prefix="/personalities", tags=["personalities"])


def _to_out(p: Personality) -> PersonalityOut:
    return PersonalityOut(
        id=p.id,
        name=p.name,
        description=p.description,
        system_prompt=p.system_prompt,
        params=p.params,
        is_builtin=p.is_builtin,
    )


@router.get("", response_model=list[PersonalityOut])
async def list_(db: DbSession, _: DeviceKey) -> list[PersonalityOut]:
    return [_to_out(p) for p in await personality_service.list_personalities(db)]


@router.post("", response_model=PersonalityOut)
async def create(
    payload: PersonalityCreate, db: DbSession, _: DeviceKey
) -> PersonalityOut:
    p = Personality(
        name=payload.name,
        description=payload.description,
        system_prompt=payload.system_prompt,
        params=payload.params,
        is_builtin=False,
    )
    db.add(p)
    await db.flush()
    return _to_out(p)


@router.patch("/{personality_id}", response_model=PersonalityOut)
async def update(
    personality_id: str, payload: PersonalityUpdate, db: DbSession, _: DeviceKey
) -> PersonalityOut:
    p = await db.get(Personality, personality_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Personality not found.")
    if p.is_builtin:
        raise HTTPException(status_code=400, detail="Built-in presets are read-only.")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(p, field, value)
    await db.flush()
    return _to_out(p)


@router.delete("/{personality_id}", status_code=204)
async def delete(personality_id: str, db: DbSession, _: DeviceKey) -> None:
    p = await db.get(Personality, personality_id)
    if p is None:
        return
    if p.is_builtin:
        raise HTTPException(status_code=400, detail="Built-in presets cannot be deleted.")
    await db.delete(p)
