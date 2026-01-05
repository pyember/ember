"""Capability metadata snapshots consumed by the Ember gateway.

The helpers here provide a deterministic view of provider features that the
runtime relies on for routing and feature gating. No network calls are made
so the gateway can boot deterministically in every environment.

Examples:
    >>> from ember.models.capabilities import build_provider_capabilities
    >>> build_provider_capabilities()["openai"]["tools"]
    True

"""

from __future__ import annotations

from ember.types import ProviderCapabilities

PROVIDER_ALIASES: dict[str, str] = {
    "azure_openai_east": "azure_openai",
    "azure_openai_west": "azure_openai",
}

# Canonical provider capability data grounded in upstream feature support.
# Downstream consumers may further constrain these values based on runtime
# adapter features (e.g., current adapters disable streaming until native support ships).
CANONICAL_PROVIDER_CAPABILITIES: dict[str, ProviderCapabilities] = {
    "openai": {
        "supports_streaming": True,
        "json_strict": True,
        "tools": True,
        "vision": True,
    },
    "azure_openai": {
        "supports_streaming": True,
        "json_strict": True,
        "tools": True,
        "vision": True,
    },
    "anthropic": {
        "supports_streaming": True,
        "json_strict": True,
        "tools": True,
        "vision": True,
    },
    "google": {
        "supports_streaming": True,
        "json_strict": True,
        "tools": True,
        "vision": True,
    },
}


def canonical_provider_capabilities() -> dict[str, ProviderCapabilities]:
    """Return a copy of the canonical capability map including aliases."""

    base = {key: value.copy() for key, value in CANONICAL_PROVIDER_CAPABILITIES.items()}
    for alias, canonical in PROVIDER_ALIASES.items():
        if canonical in base:
            base[alias] = base[canonical].copy()
    return base


def build_provider_capabilities() -> dict[str, ProviderCapabilities]:
    """Build a deterministic capability mapping for known providers.

    The result only contains fields that the gateway consumes today.

    Returns:
        dict[str, ProviderCapabilities]: Capability metadata keyed by provider slug.

    Examples:
        >>> caps = build_provider_capabilities()
        >>> caps["openai"]["supports_streaming"]
        True
    """
    return canonical_provider_capabilities()
