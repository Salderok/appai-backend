"""Voice routes: speech-to-text + text-to-speech."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.core.exceptions import AppError
from app.deps import DeviceKey
from app.voice.stt import transcribe
from app.voice.tts import synthesize

router = APIRouter(prefix="/voice", tags=["voice"])


class TranscriptionResponse(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    voice: str | None = None


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(_: DeviceKey, file: UploadFile = File(...)) -> TranscriptionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file.")
    suffix = Path(file.filename).suffix or ".m4a"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        text = await transcribe(tmp_path)
    except AppError:
        raise
    finally:
        tmp_path.unlink(missing_ok=True)
    return TranscriptionResponse(text=text)


@router.post("/synthesize")
async def synthesize_speech(payload: TTSRequest, _: DeviceKey) -> Response:
    audio = await synthesize(payload.text, voice=payload.voice)
    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'inline; filename="speech.mp3"'},
    )
