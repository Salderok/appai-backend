"""HTTP fetch tool. Lets the agent retrieve a URL and read its text content."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from app.agents.tools.base import BaseTool, ToolError

MAX_CONTENT_BYTES = 200_000


class HttpFetchArgs(BaseModel):
    url: str = Field(description="Absolute http(s) URL to fetch.")
    method: str = Field(default="GET", description="HTTP method.")


class HttpFetchTool(BaseTool):
    name = "http_fetch"
    description = (
        "Fetch a URL and return up to 200 KB of decoded body. Use for reading "
        "documentation pages, API docs, or any text resource."
    )
    args_model = HttpFetchArgs

    async def run(self, args: HttpFetchArgs) -> dict:
        if not args.url.startswith(("http://", "https://")):
            raise ToolError("Only http(s) URLs are allowed.")
        try:
            async with httpx.AsyncClient(
                timeout=15.0, follow_redirects=True, headers={"User-Agent": "appAi/0.1"}
            ) as client:
                resp = await client.request(args.method.upper(), args.url)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Fetch failed: {exc}") from exc

        body = resp.content[:MAX_CONTENT_BYTES]
        try:
            text = body.decode(resp.encoding or "utf-8", errors="replace")
        except LookupError:
            text = body.decode("utf-8", errors="replace")

        return {
            "status": resp.status_code,
            "content_type": resp.headers.get("content-type"),
            "url": str(resp.url),
            "body": text,
            "truncated": len(resp.content) > MAX_CONTENT_BYTES,
        }
