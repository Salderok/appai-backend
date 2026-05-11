"""File upload + retrieval routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy import select

from app.config import settings
from app.core.exceptions import AppError
from app.db.models.file import UploadedFile
from app.deps import DbSession, DeviceKey
from app.files.extractor import extract_text
from app.files.storage import save_bytes
from app.schemas.files import FilePreview, UploadedFileOut

router = APIRouter(prefix="/files", tags=["files"])


@router.post("", response_model=UploadedFileOut)
async def upload_file(
    db: DbSession,
    _: DeviceKey,
    file: UploadFile = File(...),
    conversation_id: str | None = Form(default=None),
) -> UploadedFileOut:
    raw = await file.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    path = save_bytes(file.filename, raw)
    mime = file.content_type or "application/octet-stream"

    extracted: str | None = None
    try:
        extracted = extract_text(path, mime)
    except AppError:
        extracted = None
    except Exception:  # noqa: BLE001 — extraction is best-effort
        extracted = None

    record = UploadedFile(
        conversation_id=conversation_id,
        filename=file.filename,
        mime_type=mime,
        size_bytes=len(raw),
        storage_path=str(path),
        extracted_text=extracted,
    )
    db.add(record)
    await db.flush()

    return UploadedFileOut(
        id=record.id,
        filename=record.filename,
        mime_type=record.mime_type,
        size_bytes=record.size_bytes,
        conversation_id=record.conversation_id,
        has_text=bool(record.extracted_text),
        created_at=record.created_at.isoformat(),
    )


@router.get("/{file_id}", response_model=FilePreview)
async def get_file(file_id: str, db: DbSession, _: DeviceKey) -> FilePreview:
    f = await db.get(UploadedFile, file_id)
    if f is None:
        raise HTTPException(status_code=404, detail="File not found.")
    return FilePreview(
        id=f.id,
        filename=f.filename,
        mime_type=f.mime_type,
        extracted_text=f.extracted_text,
    )


@router.get("", response_model=list[UploadedFileOut])
async def list_files(
    db: DbSession, _: DeviceKey, conversation_id: str | None = None
) -> list[UploadedFileOut]:
    stmt = select(UploadedFile).order_by(UploadedFile.created_at.desc()).limit(100)
    if conversation_id:
        stmt = stmt.where(UploadedFile.conversation_id == conversation_id)
    rows = list((await db.execute(stmt)).scalars().all())
    return [
        UploadedFileOut(
            id=f.id,
            filename=f.filename,
            mime_type=f.mime_type,
            size_bytes=f.size_bytes,
            conversation_id=f.conversation_id,
            has_text=bool(f.extracted_text),
            created_at=f.created_at.isoformat(),
        )
        for f in rows
    ]


@router.delete("/{file_id}", status_code=204)
async def delete_file(file_id: str, db: DbSession, _: DeviceKey) -> None:
    f = await db.get(UploadedFile, file_id)
    if f is None:
        return
    Path(f.storage_path).unlink(missing_ok=True)
    await db.delete(f)
