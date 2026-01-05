"""Protocol definitions for provider-backed model discovery."""

from __future__ import annotations

from typing import Iterable, Optional, Protocol

from ember.models.discovery.types import DiscoveredModel


class DiscoveryProvider(Protocol):
    """Minimal interface a discovery adapter must implement."""

    name: str  # canonical provider slug (e.g., "google", "openai")

    def list_models(
        self,
        *,
        region: Optional[str] = None,
        project_hint: Optional[str] = None,
    ) -> Iterable[DiscoveredModel]:
        """Return models surfaced by this provider."""


__all__ = ["DiscoveryProvider"]
