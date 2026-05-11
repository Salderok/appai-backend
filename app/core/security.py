"""Single-user device-key auth.

The mobile app stores a long random `DEVICE_KEY` in Android Keystore and
includes it as the `X-Device-Key` header on every request. The backend
compares it (constant-time) against the configured value.
"""

from __future__ import annotations

import hmac

from fastapi import Header

from app.config import settings
from app.core.exceptions import AuthError


async def require_device_key(
    x_device_key: str | None = Header(default=None, alias="X-Device-Key"),
) -> str:
    if not x_device_key:
        raise AuthError("Missing X-Device-Key header.")
    if not hmac.compare_digest(x_device_key, settings.device_key):
        raise AuthError("Invalid device key.")
    return x_device_key
