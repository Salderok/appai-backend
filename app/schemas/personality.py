"""Personality schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PersonalityOut(BaseModel):
    id: str
    name: str
    description: str | None = None
    system_prompt: str
    params: dict[str, Any] | None = None
    is_builtin: bool


class PersonalityCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = None
    system_prompt: str = Field(min_length=1)
    params: dict[str, Any] | None = None


class PersonalityUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    params: dict[str, Any] | None = None
