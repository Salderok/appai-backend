"""Web search tool.

Uses DuckDuckGo's HTML endpoint via httpx — no API key required, no SDK
dependency. The result is a small list of {title, url, snippet}. Quality is
sufficient for casual research; swap for Tavily / Brave / Serper later if
you have keys.
"""

from __future__ import annotations

import re
from html import unescape

import httpx
from pydantic import BaseModel, Field

from app.agents.tools.base import BaseTool, ToolError

DDG_URL = "https://html.duckduckgo.com/html/"
RESULT_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
    r'(?:<a[^>]*class="result__snippet"[^>]*>(.*?)</a>)?',
    re.DOTALL,
)
TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(s: str) -> str:
    return unescape(TAG_RE.sub("", s)).strip()


class WebSearchArgs(BaseModel):
    query: str = Field(description="Search query.")
    max_results: int = Field(default=5, ge=1, le=10)


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    args_model = WebSearchArgs

    async def run(self, args: WebSearchArgs) -> list[dict]:
        try:
            async with httpx.AsyncClient(
                timeout=15.0,
                headers={"User-Agent": "Mozilla/5.0 appAi/0.1"},
                follow_redirects=True,
            ) as client:
                resp = await client.post(DDG_URL, data={"q": args.query})
                resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Web search failed: {exc}") from exc

        matches = RESULT_RE.findall(resp.text)
        results: list[dict] = []
        for url, title_html, snippet_html in matches[: args.max_results]:
            results.append({
                "title": _strip_html(title_html),
                "url": url,
                "snippet": _strip_html(snippet_html or ""),
            })
        return results
