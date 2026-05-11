"""Extract plain text from uploaded files. Phase 3 wires in image OCR."""

from __future__ import annotations

from pathlib import Path

from app.core.exceptions import AppError


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise AppError("PDF support not installed. Add 'pypdf' to backend deps.") from exc
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    try:
        from docx import Document  # type: ignore
    except ImportError as exc:
        raise AppError("DOCX support not installed. Add 'python-docx' to backend deps.") from exc
    return "\n".join(p.text for p in Document(str(path)).paragraphs)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def is_image(path: Path, mime_type: str | None = None) -> bool:
    if path.suffix.lower() in IMAGE_EXTS:
        return True
    return (mime_type or "").lower().startswith("image/")


def extract_text(path: Path, mime_type: str | None = None) -> str:
    suffix = path.suffix.lower()
    mime = (mime_type or "").lower()
    if suffix == ".pdf" or mime == "application/pdf":
        return _read_pdf(path)
    if suffix in {".docx"} or "wordprocessingml" in mime:
        return _read_docx(path)
    if suffix in {".txt", ".md", ".csv", ".json", ".log"} or mime.startswith("text/"):
        return _read_text(path)
    if is_image(path, mime_type):
        # Images don't have extractable text; vision is handled at the provider level.
        return ""
    raise AppError(f"Unsupported file type for extraction: {suffix or mime}")
