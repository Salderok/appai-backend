"""File upload schemas."""

from __future__ import annotations

from pydantic import BaseModel


class UploadedFileOut(BaseModel):
    id: str
    filename: str
    mime_type: str
    size_bytes: int
    conversation_id: str | None = None
    has_text: bool
    created_at: str


class FilePreview(BaseModel):
    id: str
    filename: str
    mime_type: str
    extracted_text: str | None
