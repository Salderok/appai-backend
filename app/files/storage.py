"""Disk-backed blob storage for uploaded files.

We keep this simple — files live under `settings.upload_dir/<uuid>-<filename>`.
Phase 5 can swap this for S3-compatible storage without touching callers.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from app.config import settings


def ensure_upload_dir() -> Path:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings.upload_dir


def save_bytes(filename: str, data: bytes) -> Path:
    upload_dir = ensure_upload_dir()
    safe_name = filename.replace("/", "_").replace("\\", "_")[:120]
    path = upload_dir / f"{uuid.uuid4().hex[:12]}-{safe_name}"
    path.write_bytes(data)
    return path
