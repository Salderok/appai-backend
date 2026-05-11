"""Retry helpers for LLM provider calls.

Used by the chat service to recover from transient upstream failures.
Non-retryable errors (auth, 4xx) raise immediately.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.exceptions import ProviderError

T = TypeVar("T")


def _is_retryable(exc: BaseException) -> bool:
    if not isinstance(exc, ProviderError):
        return False
    msg = str(exc).lower()
    # Don't retry auth / quota / 4xx-y errors.
    bad = ("unauthorized", "forbidden", "invalid", "not found", "bad request", "quota")
    return not any(b in msg for b in bad)


async def call_with_retry(fn: Callable[[], Awaitable[T]], *, attempts: int = 3) -> T:
    """Invoke `fn` with exponential backoff on transient ProviderErrors."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    ):
        with attempt:
            return await fn()
    raise RuntimeError("unreachable")
