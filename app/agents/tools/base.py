"""Tool base class for the agent system.

A tool is a callable that the LLM can invoke through function-calling.
Each tool advertises a JSON schema and an async `run(args)` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolError(Exception):
    """Raised when a tool execution fails in a recoverable, user-visible way."""


class BaseTool(ABC):
    name: str
    description: str
    args_model: type[BaseModel]

    @abstractmethod
    async def run(self, args: BaseModel) -> Any: ...

    # ---- helpers used by the orchestrator --------------------------------
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }

    async def invoke(self, raw_args: dict[str, Any]) -> Any:
        args = self.args_model.model_validate(raw_args)
        return await self.run(args)
