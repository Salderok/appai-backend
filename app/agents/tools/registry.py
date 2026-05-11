"""Registry of agent tools.

Tools come in two flavors:
  - stateless (calculator, http_fetch, web_search) — instantiated once.
  - stateful  (notes, recall_memory) — need a DB session, built per-request.

`build_tools(db)` returns the full set for a given request scope.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.tools.base import BaseTool
from app.agents.tools.calculator import CalculatorTool
from app.agents.tools.http_fetch import HttpFetchTool
from app.agents.tools.notes import ListNotesTool, RecallMemoryTool, SaveNoteTool
from app.agents.tools.web_search import WebSearchTool


def build_tools(db: AsyncSession) -> list[BaseTool]:
    return [
        CalculatorTool(),
        WebSearchTool(),
        HttpFetchTool(),
        SaveNoteTool(db),
        ListNotesTool(db),
        RecallMemoryTool(db),
    ]


def tool_schemas(db: AsyncSession) -> list[dict]:
    return [tool.schema() for tool in build_tools(db)]


def get_tool(name: str, db: AsyncSession) -> BaseTool | None:
    for tool in build_tools(db):
        if tool.name == name:
            return tool
    return None
