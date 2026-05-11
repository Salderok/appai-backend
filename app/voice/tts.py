"""Text-to-speech. Returns audio bytes (MP3) for the client to play."""

from __future__ import annotations

from app.config import settings
from app.core.exceptions import ConfigError, ProviderError
from app.llm.registry import get_provider


async def synthesize(text: str, voice: str | None = None) -> bytes:
    voice = voice or settings.tts_voice

    if settings.tts_provider == "openai":
        provider = get_provider("openai")
        try:
            resp = await provider.client.audio.speech.create(  # type: ignore[attr-defined]
                model="gpt-4o-mini-tts",
                voice=voice,
                input=text,
                response_format="mp3",
            )
            return await resp.aread()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenAI TTS error: {exc}") from exc

    if settings.tts_provider == "edge":
        try:
            import edge_tts  # type: ignore
        except ImportError as exc:
            raise ConfigError(
                "Edge TTS requested but edge-tts not installed. "
                "pip install 'appai-backend[voice]'"
            ) from exc
        communicate = edge_tts.Communicate(text, voice=voice or "en-US-AriaNeural")
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    raise ConfigError(f"Unsupported TTS_PROVIDER: {settings.tts_provider}")
