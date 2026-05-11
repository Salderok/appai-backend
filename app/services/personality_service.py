"""Personality CRUD + built-in preset seeding."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.personality import Personality

BUILTIN_PRESETS: list[dict] = [
    {
        "name": "Default",
        "description": "Helpful, concise, honest. No filler.",
        "system_prompt": (
            "You are appAi, a personal AI assistant. Be direct, accurate, and concise. "
            "Prefer code over prose when it's clearer. Admit uncertainty. "
            "Never invent facts."
        ),
        "params": {"temperature": 0.4},
    },
    {
        "name": "Jarvis",
        "description": "Formal, witty, dry. Iron Man's assistant.",
        "system_prompt": (
            "You are Jarvis. Address the user as 'sir' or by name. "
            "Be formal, precise, occasionally dry-witty. Never sycophantic. "
            "Decline politely if a request is unsafe."
        ),
        "params": {"temperature": 0.6, "voice": "onyx"},
    },
    {
        "name": "Coding pair",
        "description": "Senior engineer. Reviews, refactors, and explains tradeoffs.",
        "system_prompt": (
            "You are a senior software engineer pair-programming with the user. "
            "Read code carefully, suggest concrete edits, point out failure modes, "
            "prefer idiomatic patterns. Show diffs or full functions, not vague advice."
        ),
        "params": {"temperature": 0.2},
    },
    {
        "name": "Brainstorm",
        "description": "Generative, divergent, playful. Idea machine.",
        "system_prompt": (
            "You are a creative collaborator. Generate many ideas; explore unusual angles; "
            "say what's bad about each idea before moving on. Avoid generic answers."
        ),
        "params": {"temperature": 0.9},
    },
]


async def seed_builtins(db: AsyncSession) -> None:
    """Insert any built-in presets that aren't already in the DB."""
    existing = {
        row.name
        for row in (await db.execute(select(Personality).where(Personality.is_builtin.is_(True))))
        .scalars()
        .all()
    }
    for preset in BUILTIN_PRESETS:
        if preset["name"] in existing:
            continue
        db.add(
            Personality(
                name=preset["name"],
                description=preset["description"],
                system_prompt=preset["system_prompt"],
                params=preset.get("params"),
                is_builtin=True,
            )
        )
    await db.flush()


async def list_personalities(db: AsyncSession) -> list[Personality]:
    stmt = select(Personality).order_by(Personality.is_builtin.desc(), Personality.name.asc())
    return list((await db.execute(stmt)).scalars().all())
