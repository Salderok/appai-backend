"""Chat endpoints: streaming SSE + non-streaming JSON."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from app.deps import DbSession, DeviceKey, MemoryDep
from app.schemas.chat import ChatMessageOut, ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    db: DbSession,
    memory: MemoryDep,
    _: DeviceKey,
) -> ChatResponse:
    """Non-streaming chat. Returns the full assistant reply."""
    service = await ChatService.from_request(db, memory, payload)
    completion = await service.complete(payload.messages)

    from app.services.conversation_service import list_messages

    msgs = await list_messages(db, service.conversation.id)
    assistant = next((m for m in reversed(msgs) if m.role == "assistant"), None)
    assert assistant is not None

    return ChatResponse(
        conversation_id=service.conversation.id,
        message=ChatMessageOut(
            id=assistant.id,
            role="assistant",
            content=assistant.content,
            metadata=assistant.extra,
            created_at=assistant.created_at.isoformat(),
        ),
        usage=completion.usage,
    )


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    db: DbSession,
    memory: MemoryDep,
    _: DeviceKey,
) -> EventSourceResponse:
    """Stream the assistant's reply over SSE.

    Event types emitted:
      - start: {conversation_id}
      - delta: {content}
      - done:  {finish_reason, usage, message_id}
      - error: {message}
    """
    service = await ChatService.from_request(db, memory, payload)

    async def event_source():
        try:
            async for event_name, data in service.stream(payload.messages):
                if await request.is_disconnected():
                    break
                yield {"event": event_name, "data": json.dumps(data)}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}

    return EventSourceResponse(event_source())
