"""Intermediate extraction schemas for grounded source snippets.

These models validate only the shape and quote-presence contract for a small
source-grounded extraction pass. They do not replace Samsung's official schemas
or claim semantic completeness or safety.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class GroundedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    source_quote: str

    @field_validator("text", "source_quote")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        if value is None or not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[GroundedItem]
    warnings: list[GroundedItem]
    conditions: list[GroundedItem]
    missing_details: list[GroundedItem]
