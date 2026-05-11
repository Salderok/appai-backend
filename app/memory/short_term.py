"""Short-term conversation memory: rolling buffer + on-demand summarization."""

from __future__ import annotations

from app.db.models.message import Message

MAX_BUFFER_TOKENS = 4_000  # rough cap before we summarize older turns


def estimate_tokens(text: str) -> int:
    """Crude tokens-per-char heuristic. Replace with tiktoken in Phase 2 if needed."""
    return max(1, len(text) // 4)


def select_recent(messages: list[Message], budget: int = MAX_BUFFER_TOKENS) -> list[Message]:
    """Walk newest-first and keep messages until we exhaust the token budget."""
    keep: list[Message] = []
    used = 0
    for m in reversed(messages):
        cost = estimate_tokens(m.content)
        if used + cost > budget:
            break
        keep.append(m)
        used += cost
    return list(reversed(keep))
