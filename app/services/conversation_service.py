"""CRUD operations on conversations and messages."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.db.models.conversation import Conversation
from app.db.models.message import Message


async def list_conversations(db: AsyncSession, limit: int = 50) -> list[Conversation]:
    stmt = select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit)
    return list((await db.execute(stmt)).scalars().all())


async def get_conversation(db: AsyncSession, conversation_id: str) -> Conversation:
    convo = await db.get(Conversation, conversation_id)
    if convo is None:
        raise NotFoundError(f"Conversation {conversation_id} not found.")
    return convo


async def get_or_create_conversation(
    db: AsyncSession,
    *,
    conversation_id: str | None,
    provider: str,
    model: str,
    system_prompt: str | None,
    personality_id: str | None = None,
    title: str | None = None,
) -> Conversation:
    if conversation_id:
        return await get_conversation(db, conversation_id)
    convo = Conversation(
        title=title or "New chat",
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        personality_id=personality_id,
    )
    db.add(convo)
    await db.flush()
    return convo


async def append_message(
    db: AsyncSession,
    *,
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> Message:
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        extra=metadata,
    )
    db.add(msg)
    await db.flush()
    return msg


async def message_count(db: AsyncSession, conversation_id: str) -> int:
    stmt = select(func.count()).select_from(Message).where(Message.conversation_id == conversation_id)
    return int((await db.execute(stmt)).scalar_one())


async def list_messages(db: AsyncSession, conversation_id: str) -> list[Message]:
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def delete_conversation(db: AsyncSession, conversation_id: str) -> None:
    convo = await get_conversation(db, conversation_id)
    await db.delete(convo)


async def update_conversation_title_if_empty(
    db: AsyncSession, conversation_id: str, first_user_message: str
) -> None:
    """Set a quick provisional title from the first user line.

    A better LLM-generated title may be written later by `set_title_from_llm`.
    """
    convo = await get_conversation(db, conversation_id)
    if convo.title and convo.title != "New chat":
        return
    title = first_user_message.strip().splitlines()[0][:80]
    convo.title = title or "New chat"


async def set_title(db: AsyncSession, conversation_id: str, title: str) -> None:
    convo = await get_conversation(db, conversation_id)
    convo.title = title[:120]
    await db.flush()
