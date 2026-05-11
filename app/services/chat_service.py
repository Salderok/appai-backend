"""Chat orchestration: builds the message context (history + memory + system prompt)
and delegates generation to a provider via the LLM registry.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.exceptions import ProviderError
from app.core.logging import get_logger
from app.db.models.conversation import Conversation
from app.db.models.message import Message
from app.llm.base import BaseLLMProvider
from app.llm.registry import get_provider
from app.llm.retry import call_with_retry
from app.llm.types import ChatChunk, ChatCompletion, ChatMessage, ChatOptions, ImagePart
from app.memory.manager import MemoryManager
from app.schemas.chat import ChatMessageIn, ChatRequest
from app.services import conversation_service
from app.services.titling import generate_title

logger = get_logger(__name__)


def _maybe_fallback_provider() -> BaseLLMProvider | None:
    """Return the configured Ollama provider if offline fallback is enabled."""
    if not settings.enable_offline_fallback or not settings.ollama_base_url:
        return None
    try:
        return get_provider("ollama")
    except Exception:  # noqa: BLE001
        return None


class ChatService:
    """Stateless orchestrator. Construct per-request via `ChatService.from_request`."""

    def __init__(
        self,
        db: AsyncSession,
        memory: MemoryManager,
        provider: BaseLLMProvider,
        options: ChatOptions,
        conversation: Conversation,
        system_prompt: str | None,
        use_memory: bool,
        attachments: list[str] | None = None,
    ) -> None:
        self.db = db
        self.memory = memory
        self.provider = provider
        self.options = options
        self.conversation = conversation
        self.system_prompt = system_prompt
        self.use_memory = use_memory
        self.attachments = attachments or []

    # ---- factory ----------------------------------------------------------
    @classmethod
    async def from_request(
        cls,
        db: AsyncSession,
        memory: MemoryManager,
        req: ChatRequest,
    ) -> ChatService:
        provider_id = req.provider or settings.default_provider
        model = req.model or settings.default_model
        provider = get_provider(provider_id)

        # If the requested model isn't in this provider, fall back to its first one
        # so a stale mobile cache can't crash the request.
        if model not in provider.list_models():
            model = provider.list_models()[0]

        convo = await conversation_service.get_or_create_conversation(
            db,
            conversation_id=req.conversation_id,
            provider=provider.id,
            model=model,
            system_prompt=req.system_prompt,
            personality_id=req.personality_id,
        )

        options = ChatOptions(
            model=model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        return cls(
            db=db,
            memory=memory,
            provider=provider,
            options=options,
            conversation=convo,
            system_prompt=req.system_prompt or convo.system_prompt,
            use_memory=req.use_memory,
            attachments=req.attachments,
        )

    # ---- context building -------------------------------------------------
    async def _build_messages(self, new_user_messages: list[ChatMessageIn]) -> list[ChatMessage]:
        msgs: list[ChatMessage] = []

        if self.system_prompt:
            msgs.append(ChatMessage(role="system", content=self.system_prompt))

        if self.use_memory and new_user_messages:
            latest = new_user_messages[-1].content
            memory_context = await self.memory.build_context(latest)
            if memory_context:
                msgs.append(ChatMessage(role="system", content=memory_context))

        attachment_context = await self._build_attachment_context()
        if attachment_context:
            msgs.append(ChatMessage(role="system", content=attachment_context))

        history = await conversation_service.list_messages(self.db, self.conversation.id)
        for m in history:
            msgs.append(ChatMessage(role=m.role, content=m.content))  # type: ignore[arg-type]

        images = await self._collect_attachment_images()
        for i, m in enumerate(new_user_messages):
            attach_images = images if (i == len(new_user_messages) - 1) else []
            msgs.append(
                ChatMessage(role=m.role, content=m.content, name=m.name, images=attach_images)
            )

        return msgs

    async def _collect_attachment_images(self) -> list[ImagePart]:
        if not self.attachments:
            return []
        from base64 import b64encode
        from pathlib import Path

        from app.db.models.file import UploadedFile
        from app.files.extractor import is_image

        images: list[ImagePart] = []
        for file_id in self.attachments[:4]:  # cap image count
            f = await self.db.get(UploadedFile, file_id)
            if not f:
                continue
            path = Path(f.storage_path)
            if not path.exists() or not is_image(path, f.mime_type):
                continue
            data = path.read_bytes()
            uri = f"data:{f.mime_type};base64,{b64encode(data).decode('ascii')}"
            images.append(ImagePart(data_uri=uri))
        return images

    async def _build_attachment_context(self) -> str | None:
        """Inline extracted text from any attached files as a system message.

        Vision-capable models can also see images directly — that's handled in
        the provider layer (Phase 3+) via message metadata; here we just keep
        the text path simple and provider-agnostic.
        """
        if not self.attachments:
            return None
        from app.db.models.file import UploadedFile

        parts: list[str] = []
        for file_id in self.attachments[:5]:  # cap to avoid runaway prompts
            f = await self.db.get(UploadedFile, file_id)
            if not f or not f.extracted_text:
                continue
            snippet = f.extracted_text.strip()
            if len(snippet) > 8000:
                snippet = snippet[:8000] + "\n…[truncated]"
            parts.append(f"=== Attached file: {f.filename} ({f.mime_type}) ===\n{snippet}")
        if not parts:
            return None
        return "The user has attached the following files. Use them as authoritative context:\n\n" + "\n\n".join(parts)

    # ---- generation -------------------------------------------------------
    async def complete(self, new_user_messages: list[ChatMessageIn]) -> ChatCompletion:
        # Persist the user turn first.
        for m in new_user_messages:
            await conversation_service.append_message(
                self.db, conversation_id=self.conversation.id, role=m.role, content=m.content
            )
        if new_user_messages:
            await conversation_service.update_conversation_title_if_empty(
                self.db, self.conversation.id, new_user_messages[0].content
            )

        msgs = await self._build_messages(new_user_messages)
        try:
            completion = await call_with_retry(lambda: self.provider.complete(msgs, self.options))
        except ProviderError as exc:
            fallback = _maybe_fallback_provider()
            if fallback is None:
                raise
            logger.warning("primary_provider_failed_falling_back", error=str(exc), fallback=fallback.id)
            fb_options = ChatOptions(
                model=settings.ollama_default_model,
                temperature=self.options.temperature,
                max_tokens=self.options.max_tokens,
            )
            completion = await fallback.complete(msgs, fb_options)

        await conversation_service.append_message(
            self.db,
            conversation_id=self.conversation.id,
            role="assistant",
            content=completion.content,
            metadata={
                "provider": completion.provider,
                "model": completion.model,
                "finish_reason": completion.finish_reason,
                "usage": completion.usage,
            },
        )
        await self._maybe_generate_llm_title(
            user_message=new_user_messages[-1].content if new_user_messages else "",
            assistant_message=completion.content,
        )
        await self.memory.observe_turn(
            conversation_id=self.conversation.id,
            user_message=new_user_messages[-1].content if new_user_messages else "",
            assistant_message=completion.content,
        )
        return completion

    async def stream(
        self, new_user_messages: list[ChatMessageIn]
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield (event_name, payload) tuples consumed by the SSE route.

        Events:
          - "start":       {"conversation_id": ...}
          - "delta":       {"content": "..."}
          - "done":        {"finish_reason": "...", "usage": {...}, "message_id": "..."}
        """
        for m in new_user_messages:
            await conversation_service.append_message(
                self.db, conversation_id=self.conversation.id, role=m.role, content=m.content
            )
        if new_user_messages:
            await conversation_service.update_conversation_title_if_empty(
                self.db, self.conversation.id, new_user_messages[0].content
            )

        yield "start", {"conversation_id": self.conversation.id}

        msgs = await self._build_messages(new_user_messages)
        full = ""
        last_chunk: ChatChunk | None = None
        active_provider = self.provider
        active_model = self.options.model
        try:
            async for chunk in self.provider.stream(msgs, self.options):
                if chunk.delta:
                    full += chunk.delta
                    yield "delta", {"content": chunk.delta}
                last_chunk = chunk
        except ProviderError as exc:
            fallback = _maybe_fallback_provider()
            if fallback is None or full:
                # Either no fallback or we already streamed a partial answer.
                raise
            logger.warning(
                "stream_provider_failed_falling_back",
                error=str(exc),
                fallback=fallback.id,
            )
            active_provider = fallback
            active_model = settings.ollama_default_model
            fb_options = ChatOptions(
                model=active_model,
                temperature=self.options.temperature,
                max_tokens=self.options.max_tokens,
            )
            yield "delta", {"content": "[falling back to local model]\n"}
            async for chunk in fallback.stream(msgs, fb_options):
                if chunk.delta:
                    full += chunk.delta
                    yield "delta", {"content": chunk.delta}
                last_chunk = chunk

        assistant_msg: Message = await conversation_service.append_message(
            self.db,
            conversation_id=self.conversation.id,
            role="assistant",
            content=full,
            metadata={
                "provider": active_provider.id,
                "model": active_model,
                "finish_reason": last_chunk.finish_reason if last_chunk else None,
                "usage": last_chunk.usage if last_chunk else None,
            },
        )
        await self._maybe_generate_llm_title(
            user_message=new_user_messages[-1].content if new_user_messages else "",
            assistant_message=full,
        )
        await self.memory.observe_turn(
            conversation_id=self.conversation.id,
            user_message=new_user_messages[-1].content if new_user_messages else "",
            assistant_message=full,
        )
        yield "done", {
            "finish_reason": last_chunk.finish_reason if last_chunk else None,
            "usage": last_chunk.usage if last_chunk else None,
            "message_id": assistant_msg.id,
            "title": self.conversation.title,
        }

    # ---- title -----------------------------------------------------------
    async def _maybe_generate_llm_title(self, *, user_message: str, assistant_message: str) -> None:
        """If this is the first turn, replace the placeholder title with an LLM-generated one."""
        if not user_message or not assistant_message:
            return
        count = await conversation_service.message_count(self.db, self.conversation.id)
        # 2 = first user + first assistant we just wrote.
        if count > 2:
            return
        title = await generate_title(
            self.provider,
            user_message=user_message,
            assistant_message=assistant_message,
            model=self.options.model,
        )
        if title:
            await conversation_service.set_title(self.db, self.conversation.id, title)
