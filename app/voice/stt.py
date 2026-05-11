"""Speech-to-text. Phase 0 wires up OpenAI Whisper API; local faster-whisper
is available behind WHISPER_PROVIDER=local."""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.core.exceptions import ConfigError, ProviderError
from app.llm.registry import get_provider


async def transcribe(audio_path: Path) -> str:
    if settings.whisper_provider == "openai":
        provider = get_provider("openai")
        # Use the underlying client directly because audio is not part of BaseLLMProvider.
        try:
            with audio_path.open("rb") as f:
                resp = await provider.client.audio.transcriptions.create(  # type: ignore[attr-defined]
                    model="whisper-1",
                    file=f,
                )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Whisper error: {exc}") from exc
        return resp.text

    if settings.whisper_provider == "local":
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise ConfigError(
                "Local whisper requested but faster-whisper not installed. "
                "pip install 'appai-backend[voice]'"
            ) from exc
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(audio_path))
        return "".join(seg.text for seg in segments)

    raise ConfigError(f"Unknown WHISPER_PROVIDER: {settings.whisper_provider}")
