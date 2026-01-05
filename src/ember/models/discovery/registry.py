"""Registry for discovery providers."""

from __future__ import annotations

from typing import Dict

from ember.models.discovery.provider_api import DiscoveryProvider

_REGISTRY: Dict[str, DiscoveryProvider] = {}


def register_provider(provider: DiscoveryProvider) -> None:
    """Register a discovery provider implementation."""

    _REGISTRY[provider.name] = provider


def get_provider(name: str) -> DiscoveryProvider:
    """Return a previously registered discovery provider."""

    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Discovery provider '{name}' not registered. Known: {available}") from exc


def list_providers() -> list[str]:
    """Return the list of registered discovery provider names."""

    return sorted(_REGISTRY)


__all__ = ["register_provider", "get_provider", "list_providers"]
