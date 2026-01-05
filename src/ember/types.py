from __future__ import annotations

from typing import Mapping, Sequence, TypedDict

# JSON type aliases for safe, structured payloads (JWT claims, etc.)
JSONScalar = str | int | float | bool | None
# Use Sequence/Mapping to avoid invariance issues with list/dict in mypy
JSONValue = JSONScalar | Sequence["JSONValue"] | Mapping[str, "JSONValue"]


class ProviderCapabilities(TypedDict):
    supports_streaming: bool
    json_strict: bool
    tools: bool
    vision: bool


class ModelSummary(TypedDict):
    provider: str | None
    description: str
    context_window: int | None
    status: str
