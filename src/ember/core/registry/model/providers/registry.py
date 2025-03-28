"""Provider registry module for model providers.

This module maintains a registry of provider implementations and ensures proper
synchronization with the global plugin system.
"""

from typing import Any, Callable, Dict, Type, TypeVar

from ember.plugin_system import registered_providers

# Type for provider classes
ProviderType = TypeVar("ProviderType")

# Use the centralized plugin system's provider registry as the source of truth
PROVIDER_REGISTRY: Dict[str, Type[Any]] = registered_providers


def register_provider(name: str) -> Callable[[Type[ProviderType]], Type[ProviderType]]:
    """Decorator to register a provider class with the Ember framework.

    This implementation delegates to the centralized plugin system to ensure
    consistency across the framework. New code should prefer using
    `@provider` from ember.plugin_system directly.

    Args:
        name: A unique identifier for the provider.

    Returns:
        A decorator function that registers the class and returns it unchanged.

    Example:
        @register_provider("OpenAI")
        class OpenAIModel(BaseProviderModel):
            ...
    """
    from ember.plugin_system import provider

    return provider(name)
