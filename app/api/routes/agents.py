"""Agent route: streams a plan-act-observe loop over SSE."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from app.agents.orchestrator import AgentOrchestrator
from app.agents.tools.registry import build_tools
from app.config import settings
from app.deps import DbSession, DeviceKey
from app.llm.registry import get_provider
from app.llm.types import ChatMessage, ChatOptions
from app.schemas.agent import AgentRequest

router = APIRouter(prefix="/agents", tags=["agents"])


AGENT_SYSTEM = """You are an autonomous assistant with access to tools.

Process:
1. Plan briefly (one short sentence).
2. Call tools when they help. Multiple steps are fine.
3. When you have enough information, write a final answer (no more tool calls).

Rules:
- Be concise. Don't narrate every step unless useful.
- Only call tools that exist. Don't fabricate URLs or numbers.
- If a tool fails, try a different approach or explain the limitation.
"""


@router.get("/tools")
async def list_tools(db: DbSession, _: DeviceKey) -> dict:
    return {"tools": [t.schema() for t in build_tools(db)]}


@router.post("/run/stream")
async def run_agent(
    payload: AgentRequest,
    request: Request,
    db: DbSession,
    _: DeviceKey,
) -> EventSourceResponse:
    provider_id = payload.provider or settings.default_provider
    model = payload.model or settings.default_model
    provider = get_provider(provider_id)
    if model not in provider.list_models():
        model = provider.list_models()[0]

    system_text = payload.system_prompt or AGENT_SYSTEM
    messages = [
        ChatMessage(role="system", content=system_text),
        ChatMessage(role="user", content=payload.task),
    ]
    options = ChatOptions(model=model, temperature=0.2, max_tokens=2000)

    orch = AgentOrchestrator(provider, db, max_steps=payload.max_steps)

    async def event_source():
        yield {"event": "start", "data": json.dumps({"provider": provider.id, "model": model})}
        try:
            async for step in orch.run(messages, options):
                if await request.is_disconnected():
                    break
                yield {"event": step["kind"], "data": json.dumps(step)}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": json.dumps({"content": str(exc)})}
        finally:
            yield {"event": "done", "data": json.dumps({})}

    return EventSourceResponse(event_source())
