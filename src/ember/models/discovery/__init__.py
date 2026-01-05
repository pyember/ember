"""Discovery-layer primitives for provider-backed model metadata."""

from .merge import merge_catalog  # noqa: F401
from .provider_api import DiscoveryProvider  # noqa: F401
from .registry import get_provider, list_providers, register_provider  # noqa: F401
from .types import DiscoveredModel, ModelKey, OverrideSpec  # noqa: F401

__all__ = [
    "DiscoveryProvider",
    "DiscoveredModel",
    "ModelKey",
    "OverrideSpec",
    "merge_catalog",
    "get_provider",
    "list_providers",
    "register_provider",
]
