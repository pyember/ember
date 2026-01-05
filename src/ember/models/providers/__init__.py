"""Utilities for registering and resolving Ember model providers.

The registry keeps a small set of built-in providers and lets applications add their own at
runtime when needed. Provider implementations are imported lazily to keep lightweight imports
(e.g., ``import ember``) fast and avoid pulling heavyweight SDKs until required.

Examples:
    >>> from ember.models.providers import register_provider
    >>> register_provider('demo', object)  # doctest: +SKIP
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from typing import TypeAlias

from ember.models.providers.base import BaseProvider

ProviderEntry: TypeAlias = str | type[BaseProvider]

# Provider implementations are referenced by module path to avoid importing heavy SDKs at
# module import time. Entries are resolved to classes on first use and cached.
PROVIDERS: dict[str, ProviderEntry] = {
    "openai": "ember.models.providers.openai:OpenAIProvider",
    "anthropic": "ember.models.providers.anthropic:AnthropicProvider",
    "google": "ember.models.providers.google:GoogleProvider",
    "deepmind": "ember.models.providers.google:GoogleProvider",  # Alias for backward compat
    "ollama": "ember.models.providers.ollama:OllamaProvider",
}


def _dependency_available(module_name: str) -> bool:
    """Return True if the referenced dependency can be imported without loading it."""

    spec = importlib.util.find_spec(module_name)
    return spec is not None


def _config_value(path: str) -> object | None:
    from ember._internal.context.runtime import EmberContext

    value = EmberContext.current().get_config(path)
    return value


def _register_optional_provider(name: str, spec: str, *, dependency: str | None = None) -> None:
    if dependency and not _dependency_available(dependency):
        return
    PROVIDERS[name] = spec


if _dependency_available("dashscope"):
    _register_optional_provider("dashscope", "ember.models.providers.dashscope:DashScopeProvider")

if _config_value("providers.azure_openai.api_key") and _config_value(
    "providers.azure_openai.endpoint"
):
    _register_optional_provider(
        "azure_openai",
        "ember.models.providers.azure_openai:AzureOpenAIProvider",
        dependency="openai",
    )

if _config_value("providers.vertex.project"):
    _register_optional_provider(
        "vertex",
        "ember.models.providers.vertex:VertexAIProvider",
        dependency="vertexai",
    )

if _config_value("providers.bedrock.region"):
    _register_optional_provider(
        "bedrock",
        "ember.models.providers.bedrock:BedrockConverseProvider",
        dependency="boto3",
    )


# Custom providers can be registered at runtime.
# This separation allows extending without modifying core code.
_custom_providers: dict[str, type[BaseProvider]] = {}


def register_provider(name: str, provider_class: type[BaseProvider]) -> None:
    """Register a custom provider implementation.

    This function enables runtime extension of the provider system without modifying core code. It
    is designed for enterprise users who need to integrate proprietary models or custom endpoints.

    The registration is global and persists for the process lifetime. Custom providers take
    precedence over core providers with the same name.

    Args:
        name: Provider identifier. Should be lowercase, no spaces.
            Examples: "azure", "bedrock", "enterprise"
        provider_class: Class inheriting from BaseProvider. Must implement the complete provider
            interface.

    Raises:
        TypeError: If provider_class doesn't inherit from BaseProvider.
        ValueError: If name is empty.

    Examples:
        Basic registration:

        >>> from my_company import EnterpriseProvider
        >>> register_provider("enterprise", EnterpriseProvider)
        >>> response = models("enterprise/gpt-4", "Hello")

        Override core provider:

        >>> from my_azure import AzureOpenAIProvider
        >>> register_provider("openai", AzureOpenAIProvider)
        >>> # Now all OpenAI calls route through Azure

        With validation:

        >>> class CustomProvider(BaseProvider):
        ...     def complete(self, prompt, model, **kwargs):
        ...         # Custom implementation
        ...         pass
        >>> register_provider("custom", CustomProvider)
    """
    if not name:
        raise ValueError("Provider name cannot be empty")

    if not isinstance(provider_class, type):
        raise TypeError("provider_class must be a class, not an instance")

    if not issubclass(provider_class, BaseProvider):
        raise TypeError(
            f"Provider class must inherit from BaseProvider, got {provider_class.__name__}"
        )

    _custom_providers[name] = provider_class


def unregister_provider(name: str) -> bool:
    """Unregister a custom provider.

    Removes a previously registered custom provider. Core providers cannot be unregistered,
    ensuring system stability.

    Args:
        name: The provider name to unregister.

    Returns:
        True if provider was unregistered, False if not found or core.
    """
    if name in _custom_providers:
        del _custom_providers[name]
        return True
    return False


def _load_provider_class(name: str, entry: ProviderEntry) -> type[BaseProvider]:
    if isinstance(entry, str):
        if ":" not in entry:
            raise ValueError(f"Invalid provider spec for '{name}': '{entry}'")
        module_path, class_name = entry.split(":", 1)
        module = importlib.import_module(module_path)
        provider_class = getattr(module, class_name)
    else:
        provider_class = entry

    if not isinstance(provider_class, type) or not issubclass(provider_class, BaseProvider):
        raise TypeError(
            f"Provider '{name}' must resolve to a BaseProvider subclass (got {provider_class!r})"
        )

    if name in PROVIDERS and isinstance(entry, str):
        PROVIDERS[name] = provider_class
    return provider_class


def get_provider_class(name: str) -> type[BaseProvider]:
    """Get provider implementation class by name.

    This is the primary lookup function used by the registry. It checks custom providers first,
    allowing overrides of core functionality.

    Args:
        name: Provider identifier (e.g., "openai", "anthropic", "custom").

    Returns:
        The provider implementation class (not instance).

    Raises:
        ValueError: If provider name is not found.
    """
    custom_entry = _custom_providers.get(name)
    if custom_entry is not None:
        # _custom_providers only stores type[BaseProvider], not str
        return _load_provider_class(name, custom_entry)

    provider_entry: ProviderEntry | None = PROVIDERS.get(name)
    if provider_entry is None:
        available = list_providers()
        raise ValueError(
            f"Unknown provider '{name}'. Available providers: {', '.join(sorted(available))}"
        )

    return _load_provider_class(name, provider_entry)


def list_providers() -> list[str]:
    """List all available provider names."""
    all_providers = set(PROVIDERS.keys()) | set(_custom_providers.keys())
    return sorted(all_providers)


def is_provider_available(name: str) -> bool:
    """Check if a provider is available."""
    return name in _custom_providers or name in PROVIDERS


def get_provider_info(name: str) -> dict[str, str]:
    """Get information about a provider."""
    provider_class = get_provider_class(name)
    provider_type = "custom" if name in _custom_providers else "core"
    return {
        "name": name,
        "type": provider_type,
        "class": provider_class.__name__,
        "module": provider_class.__module__,
    }


def resolve_model_id(model: str) -> tuple[str, str]:
    """Resolve a model string to (provider, model_name).

    This function implements the model resolution logic that makes Ember's API intuitive. It
    handles both explicit notation (provider/model) and implicit notation where the provider is
    inferred from model naming conventions.

    Resolution Rules:
        1. If "/" present: Split on first "/" as provider/model
        2. Otherwise, infer from prefixes:
           - gpt-*, davinci, etc. -> openai
           - claude* -> anthropic
           - gemini* -> google
           - Others -> "unknown"
    """
    # Check for explicit provider notation
    if "/" in model:
        parts = model.split("/", 1)
        provider, mname = parts[0], parts[1]
        # Guarded alias: allow local/* to route to ollama when enabled
        if provider == "local" and os.getenv("EMBER_LOCAL_ALIAS", "0") in {"1", "true", "True"}:
            return "ollama", mname
        return provider, mname

    # Infer provider from well-known model naming patterns.
    # This list is intentionally conservative - only models with
    # clear provider association are mapped.
    model_lower = model.lower()

    # Special-case: plain 'ollama' selects default local model
    if model_lower == "ollama":
        return "ollama", os.getenv("EMBER_OLLAMA_DEFAULT_MODEL", "auto")

    # OpenAI models - comprehensive list of known patterns
    if (
        model_lower.startswith("gpt-")
        or model_lower.startswith("davinci")
        or model_lower.startswith("babbage")
        or model_lower.startswith("ada")
        or model_lower.startswith("text-")
        or model_lower.startswith("o1-")  # Reasoning models
        or model_lower.startswith("o3-")  # Reasoning models
        or model_lower.startswith("o4-")  # Reasoning models
        or model_lower.startswith("codex-")
        or model_lower.startswith("computer-use-")
        or model_lower
        in [
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
            "o1-pro",
            "o1-mini",
            "o3",
            "o3-pro",
            "o3-mini",
            "o4-mini",
        ]
    ):
        return "openai", model

    # Anthropic models - all Claude variants
    elif model_lower.startswith("claude"):
        return "anthropic", model

    # Google models - Gemini family
    elif model_lower.startswith("gemini") or model_lower.startswith("models/gemini"):
        return "google", model

    # Models without clear provider association return "unknown".
    # This includes Llama, Mistral, etc. which can be served by
    # multiple providers. The registry will provide a better error.
    return "unknown", model


__all__ = [
    "register_provider",
    "unregister_provider",
    "get_provider_class",
    "list_providers",
    "is_provider_available",
    "get_provider_info",
    "resolve_model_id",
    "PROVIDERS",
]
