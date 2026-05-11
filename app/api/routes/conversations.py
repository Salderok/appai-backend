"""CRUD over conversations."""

from __future__ import annotations

from fastapi import APIRouter

from app.deps import DbSession, DeviceKey
from app.schemas.conversation import (
    ConversationCreate,
    ConversationDetail,
    ConversationSummary,
    ConversationUpdate,
    MessageOut,
)
from app.services import conversation_service

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationSummary])
async def list_(db: DbSession, _: DeviceKey) -> list[ConversationSummary]:
    convos = await conversation_service.list_conversations(db)
    out: list[ConversationSummary] = []
    for c in convos:
        out.append(
            ConversationSummary(
                id=c.id,
                title=c.title,
                provider=c.provider,
                model=c.model,
                created_at=c.created_at.isoformat(),
                updated_at=c.updated_at.isoformat(),
                message_count=await conversation_service.message_count(db, c.id),
            )
        )
    return out


@router.post("", response_model=ConversationSummary)
async def create(payload: ConversationCreate, db: DbSession, _: DeviceKey) -> ConversationSummary:
    convo = await conversation_service.get_or_create_conversation(
        db,
        conversation_id=None,
        provider=payload.provider or "openai",
        model=payload.model or "gpt-4o-mini",
        system_prompt=payload.system_prompt,
        personality_id=payload.personality_id,
        title=payload.title,
    )
    return ConversationSummary(
        id=convo.id,
        title=convo.title,
        provider=convo.provider,
        model=convo.model,
        created_at=convo.created_at.isoformat(),
        updated_at=convo.updated_at.isoformat(),
        message_count=0,
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_one(conversation_id: str, db: DbSession, _: DeviceKey) -> ConversationDetail:
    convo = await conversation_service.get_conversation(db, conversation_id)
    messages = await conversation_service.list_messages(db, conversation_id)
    return ConversationDetail(
        id=convo.id,
        title=convo.title,
        provider=convo.provider,
        model=convo.model,
        system_prompt=convo.system_prompt,
        created_at=convo.created_at.isoformat(),
        updated_at=convo.updated_at.isoformat(),
        message_count=len(messages),
        messages=[
            MessageOut(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at.isoformat(),
                metadata=m.extra,
            )
            for m in messages
        ],
    )


@router.patch("/{conversation_id}", response_model=ConversationSummary)
async def update(
    conversation_id: str, payload: ConversationUpdate, db: DbSession, _: DeviceKey
) -> ConversationSummary:
    convo = await conversation_service.get_conversation(db, conversation_id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(convo, field, value)
    await db.flush()
    return ConversationSummary(
        id=convo.id,
        title=convo.title,
        provider=convo.provider,
        model=convo.model,
        created_at=convo.created_at.isoformat(),
        updated_at=convo.updated_at.isoformat(),
        message_count=await conversation_service.message_count(db, convo.id),
    )


@router.delete("/{conversation_id}", status_code=204)
async def delete(conversation_id: str, db: DbSession, _: DeviceKey) -> None:
    await conversation_service.delete_conversation(db, conversation_id)
