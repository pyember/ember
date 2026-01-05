from ember.models.discovery.registry import register_provider as register_discovery_provider

from ._discovery import OpenAIDiscoveryAdapter
from .provider import OpenAIProvider

__all__ = ["OpenAIProvider", "OpenAIDiscoveryAdapter"]

register_discovery_provider(OpenAIDiscoveryAdapter())
