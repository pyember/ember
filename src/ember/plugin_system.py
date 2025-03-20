"""Plugin Registration System for Ember Model Providers.

Providers (e.g., OpenAI, Anthropic, etc.) can register themselves using the
@provider decorator. This mechanism decouples provider implementations from the core Ember code.
"""

from typing import Any, Callable, Dict, Type

# Global registry mapping provider names to their corresponding provider classes.
registered_providers: Dict[str, Type[Any]] = {}


def provider(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """Decorator to register a model provider class with Ember.

    This decorator registers the provider class in a global registry under the specified name.
    It enables provider implementations to be decoupled from core Ember logic.

    Example:
        @provider(name="OpenAI")
        class OpenAIModel(BaseProviderModel):
            ...

    Args:
        name (str): The unique provider name used for registration.

    Returns:
        Callable[[Type[Any]], Type[Any]]: A decorator that registers the provider class.
    """

    def decorator(provider_class: Type[Any]) -> Type[Any]:
        registered_providers[name] = provider_class
        return provider_class

    return decorator
