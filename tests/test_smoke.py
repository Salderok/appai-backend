"""Smoke tests: app boots, /health responds, auth gates work."""

from __future__ import annotations

import os

# Use an in-memory sqlite + dummy device key so tests don't touch real services.
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("DEVICE_KEY", "test-device-key")

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_root_ok() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.get("/")
        assert r.status_code == 200
        assert r.json()["name"] == "appAi"


@pytest.mark.asyncio
async def test_health_ok() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "providers" in body


@pytest.mark.asyncio
async def test_chat_requires_device_key() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.post("/v1/chat", json={"messages": []})
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_models_requires_device_key() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.get("/v1/models")
        assert r.status_code == 401
        r = await ac.get("/v1/models", headers={"X-Device-Key": "test-device-key"})
        assert r.status_code == 200
        assert "providers" in r.json()
