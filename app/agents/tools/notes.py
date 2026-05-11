"""Notes tool — write/read user notes via the memory store.

Internally backed by `MemoryItem` rows with `kind="note"`. Lets the agent
record things the user wants to remember without polluting the auto-extracted
fact stream.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.tools.base import BaseTool
from app.db.models.memory import MemoryItem
from app.memory.long_term import VectorMemory


class SaveNoteArgs(BaseModel):
    content: str = Field(description="The note content to save.", min_length=1, max_length=2000)
    importance: float = Field(default=0.6, ge=0.0, le=1.0)


class SaveNoteTool(BaseTool):
    name = "save_note"
    description = "Save a note for the user. Use for things they explicitly want remembered."
    args_model = SaveNoteArgs

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def run(self, args: SaveNoteArgs) -> dict:
        memory = VectorMemory(self.db)
        item = await memory.add(content=args.content, kind="note", importance=args.importance)
        return {"id": item.id, "saved": True}


class ListNotesArgs(BaseModel):
    limit: int = Field(default=20, ge=1, le=100)


class ListNotesTool(BaseTool):
    name = "list_notes"
    description = "List the user's saved notes (most recent first)."
    args_model = ListNotesArgs

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def run(self, args: ListNotesArgs) -> list[dict]:
        stmt = (
            select(MemoryItem)
            .where(MemoryItem.kind == "note")
            .order_by(MemoryItem.created_at.desc())
            .limit(args.limit)
        )
        rows = list((await self.db.execute(stmt)).scalars().all())
        return [{"id": r.id, "content": r.content, "created_at": r.created_at.isoformat()} for r in rows]


class RecallMemoryArgs(BaseModel):
    query: str = Field(description="What to look up in long-term memory.")
    k: int = Field(default=5, ge=1, le=10)


class RecallMemoryTool(BaseTool):
    name = "recall_memory"
    description = "Search the user's long-term memory by semantic similarity."
    args_model = RecallMemoryArgs

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def run(self, args: RecallMemoryArgs) -> list[dict]:
        memory = VectorMemory(self.db)
        results = await memory.search(args.query, k=args.k)
        return [
            {"content": item.content, "kind": item.kind, "score": round(score, 3)}
            for item, score in results
        ]
