"""Shared dataclasses for model discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, TypedDict


@dataclass(frozen=True)
class DiscoveredModel:
    """Canonical DTO emitted by discovery adapters."""

    provider: str
    id: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    context_window_in: Optional[int] = None
    context_window_out: Optional[int] = None
    capabilities: Sequence[str] = field(default_factory=tuple)
    region_scope: Sequence[str] = field(default_factory=tuple)
    raw_payload: Optional[Mapping[str, Any]] = None

    def model_key(self) -> str:
        """Return the normalized model key for internal maps."""

        return ModelKey.to_key(self.provider, self.id)


class OverrideSpec(TypedDict, total=False):
    """User-provided overrides applied after discovery and bootstrap merges."""

    description: Optional[str]
    pricing: Mapping[str, Any]
    hidden: Optional[bool]
    aliases: Sequence[str]
    capabilities: Sequence[str]
    context_window: Optional[int]
    context_window_out: Optional[int]


class ModelKey:
    """Helpers for working with provider-qualified model identifiers."""

    @staticmethod
    def to_key(provider: str, vendor_id: str) -> str:
        return f"{provider}:{vendor_id}"

    @staticmethod
    def split(key: str) -> tuple[str, str]:
        provider, _, vendor_id = key.partition(":")
        if not vendor_id:
            raise ValueError(f"Invalid model key '{key}'. Expected '<provider>:<id>'.")
        return provider, vendor_id


__all__ = ["DiscoveredModel", "OverrideSpec", "ModelKey"]
