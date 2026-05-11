"""Fact extractor.

After a chat turn completes, we (cheaply) ask the LLM:
  "From this exchange, list 0-5 durable facts about the user as a JSON array."

Anything returned is embedded and stored in `MemoryItem` for future retrieval.
The call is opportunistic — if it fails, the turn still completes normally.
"""

from __future__ import annotations

import json
import re

from app.core.exceptions import ProviderError
from app.core.logging import get_logger
from app.llm.base import BaseLLMProvider
from app.llm.types import ChatMessage, ChatOptions
from app.memory.long_term import VectorMemory

logger = get_logger(__name__)

FACT_SYSTEM = """You extract durable facts about the user from a single chat exchange.

Output rules:
- Return ONLY a JSON array, no prose.
- Each item: {"content": "<one short factual statement>", "kind": "fact|preference|task", "importance": 0.0-1.0}
- Include only stable, useful facts (preferences, recurring projects, identity, long-term goals).
- Skip: greetings, one-off questions, transient context, anything the user didn't actually state.
- Empty array [] is correct when nothing durable was said.
- Maximum 5 items.
"""

# Tolerate models that wrap the JSON in markdown fences.
_JSON_RE = re.compile(r"\[.*\]", re.DOTALL)


class FactExtractor:
    def __init__(self, provider: BaseLLMProvider, memory: VectorMemory, *, model: str) -> None:
        self.provider = provider
        self.memory = memory
        self.model = model

    async def extract_and_store(
        self,
        *,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
    ) -> int:
        if len(user_message) < 8:
            return 0

        prompt = (
            f"USER said: {user_message[:1500]}\n\n"
            f"ASSISTANT replied: {assistant_message[:800]}\n\n"
            "Return the JSON array of durable facts now."
        )
        try:
            completion = await self.provider.complete(
                [
                    ChatMessage(role="system", content=FACT_SYSTEM),
                    ChatMessage(role="user", content=prompt),
                ],
                ChatOptions(model=self.model, temperature=0.1, max_tokens=300),
            )
        except ProviderError as exc:
            logger.warning("fact_extraction_skipped", error=str(exc))
            return 0

        items = _parse_facts(completion.content)
        stored = 0
        for it in items:
            try:
                await self.memory.add(
                    content=it["content"],
                    kind=it.get("kind", "fact"),
                    importance=float(it.get("importance", 0.5)),
                    source_conversation_id=conversation_id,
                )
                stored += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("fact_store_failed", error=str(exc))
        if stored:
            logger.info("facts_stored", count=stored, conversation_id=conversation_id)
        return stored


def _parse_facts(raw: str) -> list[dict]:
    match = _JSON_RE.search(raw)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for entry in data[:5]:
        if not isinstance(entry, dict):
            continue
        content = (entry.get("content") or "").strip()
        if not content or len(content) < 4:
            continue
        out.append(entry)
    return out
