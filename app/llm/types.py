"""Provider-agnostic chat data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ImagePart:
    """A single image attachment on a user message.

    `data_uri` is "data:image/png;base64,..." or a remote URL. Providers
    that support vision (OpenAI gpt-4o, Anthropic 3.5+, Gemini) will receive
    it as part of the message; others see only the text content.
    """

    data_uri: str
    detail: Literal["auto", "low", "high"] = "auto"


@dataclass(slots=True)
class ChatMessage:
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    images: list[ImagePart] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatChunk:
    """A streaming delta from a provider."""

    delta: str
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    raw: Any | None = None


@dataclass(slots=True)
class ChatCompletion:
    """Full non-streaming response."""

    content: str
    finish_reason: str | None
    model: str
    provider: str
    usage: dict[str, Any] | None = None
    raw: Any | None = None


@dataclass(slots=True)
class ChatOptions:
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
