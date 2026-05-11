"""Agent orchestrator: plan → call tools → observe → respond.

Implements a minimal ReAct loop on top of the OpenAI tool-calling protocol.
Non-OpenAI providers: we fall back to a plain completion (no tools) so the
UX degrades gracefully.

Events yielded for the streaming API:
  - {"kind": "thinking",    "content": "<planner message>"}
  - {"kind": "tool_call",   "name":    "...",  "arguments": "..."}
  - {"kind": "tool_result", "name":    "...",  "result":    "..."}
  - {"kind": "final",       "content": "..."}
  - {"kind": "error",       "content": "..."}
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.tools.base import ToolError
from app.agents.tools.registry import build_tools, get_tool
from app.core.logging import get_logger
from app.llm.base import BaseLLMProvider
from app.llm.types import ChatMessage, ChatOptions

logger = get_logger(__name__)

MAX_STEPS = 6
TOOL_RESULT_MAX_CHARS = 4000


def _truncate(obj: Any) -> str:
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    if len(s) > TOOL_RESULT_MAX_CHARS:
        return s[:TOOL_RESULT_MAX_CHARS] + "\n…[truncated]"
    return s


class AgentOrchestrator:
    def __init__(
        self,
        provider: BaseLLMProvider,
        db: AsyncSession,
        *,
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.provider = provider
        self.db = db
        self.max_steps = max_steps

    async def run(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[dict[str, Any]]:
        tools = build_tools(self.db)
        tool_schemas = [t.schema() for t in tools]
        options.tools = tool_schemas

        # Only OpenAI-compatible providers support tool calls cleanly.
        if self.provider.id not in {"openai", "deepseek"}:
            try:
                completion = await self.provider.complete(messages, options)
                yield {"kind": "final", "content": completion.content}
            except Exception as exc:  # noqa: BLE001
                yield {"kind": "error", "content": str(exc)}
            return

        working = list(messages)

        for _step in range(self.max_steps):
            try:
                completion = await self.provider.complete(working, options)
            except Exception as exc:  # noqa: BLE001
                yield {"kind": "error", "content": str(exc)}
                return

            raw = completion.raw
            tool_calls = self._extract_tool_calls(raw)

            if not tool_calls:
                yield {"kind": "final", "content": completion.content}
                return

            # Assistant's "thinking" text (if any) alongside the tool calls.
            if completion.content:
                yield {"kind": "thinking", "content": completion.content}

            # Append the assistant turn carrying the tool_calls so the next
            # LLM call has the full tool-use history.
            working.append(
                ChatMessage(
                    role="assistant",
                    content=completion.content or "",
                    metadata={"tool_calls": self._serialize_tool_calls(tool_calls)},
                )
            )

            for call in tool_calls:
                name = call["name"]
                args_raw = call["arguments"] or "{}"
                yield {"kind": "tool_call", "name": name, "arguments": args_raw, "id": call["id"]}

                tool = get_tool(name, self.db)
                if tool is None:
                    result = f"Unknown tool: {name}"
                else:
                    try:
                        parsed = json.loads(args_raw) if args_raw else {}
                        result_obj = await tool.invoke(parsed)
                        result = _truncate(result_obj)
                    except ToolError as exc:
                        result = f"Tool error: {exc}"
                    except json.JSONDecodeError as exc:
                        result = f"Invalid tool arguments: {exc}"
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("tool_crashed", tool=name, error=str(exc))
                        result = f"Tool crashed: {exc}"

                yield {"kind": "tool_result", "name": name, "result": result, "id": call["id"]}

                working.append(
                    ChatMessage(
                        role="tool",
                        content=result,
                        name=name,
                        tool_call_id=call["id"],
                    )
                )

        yield {"kind": "final", "content": "(max steps reached)"}

    # ---- provider-raw plumbing (OpenAI/DeepSeek) --------------------------
    @staticmethod
    def _extract_tool_calls(raw: Any) -> list[dict[str, str]]:
        """Return [{id, name, arguments}] from an OpenAI completion response."""
        if not raw or not hasattr(raw, "choices") or not raw.choices:
            return []
        msg = raw.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        out: list[dict[str, str]] = []
        for c in calls:
            out.append({
                "id": c.id,
                "name": c.function.name,
                "arguments": c.function.arguments or "",
            })
        return out

    @staticmethod
    def _serialize_tool_calls(tool_calls: list[dict[str, str]]) -> list[dict]:
        return [
            {
                "id": t["id"],
                "type": "function",
                "function": {"name": t["name"], "arguments": t["arguments"]},
            }
            for t in tool_calls
        ]
