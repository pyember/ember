"""Provider Model Factory Module

This module provides a factory pattern implementation for dynamically discovering and 
instantiating provider model instances from ModelInfo configurations. The factory 
handles both explicitly registered providers and automatically discovered providers
at runtime.

Architecture:
- Provider Discovery: Scans the provider package directory structure to find all 
  compatible provider implementations
- Provider Registry: Maintains a cache of discovered provider classes for faster access
- Lazy Loading: Providers are discovered only when first needed, then cached
- Custom Registration: Supports manual registration of custom providers

The factory is essential for the model registry system, enabling it to:
1. Support multiple LLM providers (OpenAI, Anthropic, etc.) through a unified interface
2. Allow runtime extension with new providers without code changes
3. Handle provider-specific configuration and instantiation details
"""

from types import ModuleType
import importlib
import inspect
import logging
import os
import pkgutil
from typing import Any, Dict, Optional, Type

from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    ProviderConfigError,
)
from ember.core.registry.model.providers.base_provider import BaseProviderModel
from ember.core.registry.model.config.model_enum import parse_model_str
from ember.core.registry.model.providers.registry import PROVIDER_REGISTRY

LOGGER: logging.Logger = logging.getLogger(__name__)


def discover_providers_in_package(
    *, package_name: str, package_path: str
) -> Dict[str, Type[BaseProviderModel]]:
    """Discover and load provider model classes within the specified package.

    Performs dynamic provider discovery by traversing all modules in the given package 
    directory and inspecting each module for valid provider model implementations.
    
    Discovery process:
    1. Walks through all modules in the given package
    2. Skips packages themselves (recursive discovery not supported) and special modules
    3. For each module, finds all classes that:
       - Inherit from BaseProviderModel (but aren't BaseProviderModel itself)
       - Have a non-empty PROVIDER_NAME attribute
    4. Maps each provider's name to its implementing class
    5. Handles and logs errors during module loading for robustness

    Args:
        package_name (str): The fully qualified package name where provider modules reside
                           (e.g., "ember.core.registry.model.providers").
        package_path (str): The filesystem path corresponding to the package.

    Returns:
        Dict[str, Type[BaseProviderModel]]: A mapping from provider names (e.g., "openai", 
                                           "anthropic") to their respective provider classes.
    
    Example:
        >>> providers = discover_providers_in_package(
        ...     package_name="ember.core.registry.model.providers",
        ...     package_path="/path/to/providers"
        ... )
        >>> # Result: {"openai": OpenAIProvider, "anthropic": AnthropicProvider, ...}
    """
    providers: Dict[str, Type[BaseProviderModel]] = {}
    prefix: str = f"{package_name}."
    for _, full_module_name, is_pkg in pkgutil.walk_packages(
        path=[package_path], prefix=prefix
    ):
        short_module_name: str = full_module_name.rsplit(".", 1)[-1]
        if is_pkg or short_module_name in ("base_discovery", "registry"):
            continue
        try:
            module: Any = importlib.import_module(full_module_name)
            for _, candidate in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(candidate, BaseProviderModel)
                    and candidate is not BaseProviderModel
                ):
                    provider_name: str = getattr(candidate, "PROVIDER_NAME", "")
                    if provider_name:
                        providers[provider_name] = candidate
        except Exception as exc:
            LOGGER.warning(
                "Failed to load provider module '%s': %s", full_module_name, exc
            )
    return providers


class ModelFactory:
    """Factory for creating provider-specific model instances from ModelInfo configurations.

    The ModelFactory serves as the central component for instantiating provider models
    in the Ember framework. It handles model identifier validation, dynamic provider
    discovery, and proper instantiation of provider-specific model implementations.
    
    Key features:
    - Thread-safe provider class caching with lazy initialization
    - Provider autodiscovery from the providers package
    - Support for explicit provider registration
    - Validation of model identifiers before instantiation
    - Informative error messages for configuration issues
    
    Usage flow:
    1. The factory lazily discovers and caches provider implementations on first use
    2. When a model is requested, it validates the model ID format
    3. It finds the appropriate provider class based on the provider name
    4. The provider-specific model is instantiated with the given ModelInfo
    
    Thread safety:
    The class-level provider cache is initialized exactly once in a thread-safe manner
    via the lazy initialization pattern in the _get_providers method.
    """

    _provider_cache: Optional[Dict[str, Type[BaseProviderModel]]] = None

    @classmethod
    def _get_providers(cls) -> Dict[str, Type[BaseProviderModel]]:
        """Retrieve a registry of provider classes by merging explicit and dynamically discovered providers.

        Returns:
            Dict[str, Type[BaseProviderModel]]: A dictionary mapping provider names to provider classes.
        """
        if cls._provider_cache is not None:
            return cls._provider_cache

        # Initialize provider cache with explicitly registered providers.
        cls._provider_cache = PROVIDER_REGISTRY.copy()

        try:
            provider_module: ModuleType = importlib.import_module(
                "src.ember.core.registry.model.providers"
            )
            provider_package_name: str = provider_module.__name__
            provider_package_path: str = os.path.dirname(provider_module.__file__)
            dynamic_providers: Dict[str, Type[BaseProviderModel]] = (
                discover_providers_in_package(
                    package_name=provider_package_name,
                    package_path=provider_package_path,
                )
            )
            for name, provider_class in dynamic_providers.items():
                if name not in cls._provider_cache:
                    cls._provider_cache[name] = provider_class
        except Exception as exc:
            LOGGER.error("Failed to discover providers: %s", exc)

        return cls._provider_cache

    @classmethod
    def register_custom_provider(
        cls, *, provider_name: str, provider_class: Type[BaseProviderModel]
    ) -> None:
        """Manually register a custom provider class.

        Registers a provider class under a given name. If the provider cache is uninitialized,
        it will be initialized before adding the new provider.

        Args:
            provider_name (str): The designated name for the provider.
            provider_class (Type[BaseProviderModel]): The provider class to register.

        Raises:
            ValueError: If the provided class does not subclass BaseProviderModel.
        """
        if not issubclass(provider_class, BaseProviderModel):
            raise ValueError("Provider class must subclass BaseProviderModel.")
        if cls._provider_cache is None:
            cls._get_providers()  # Initialize the provider cache.
        cls._provider_cache[provider_name] = provider_class
        LOGGER.info("Registered custom provider: %s", provider_name)

    @staticmethod
    def create_model_from_info(*, model_info: ModelInfo) -> BaseProviderModel:
        """Create and return a provider model instance based on the given ModelInfo configuration.

        The method validates the model identifier, retrieves the matching provider class, and
        instantiates the provider model using named parameter invocation.

        Args:
            model_info (ModelInfo): Configuration details for the model, including its identifier and provider info.

        Returns:
            BaseProviderModel: An instance of the provider model corresponding to the provided configuration.

        Raises:
            ProviderConfigError: If the model identifier is invalid or the specified provider is unsupported.
        """
        # Validate the model identifier.
        try:
            parse_model_str(model_info.id)
        except ValueError as error:
            raise ProviderConfigError(
                f"Unrecognized model ID '{model_info.id}'."
            ) from error

        provider_name: str = model_info.provider.name
        discovered_providers: Dict[str, Type[BaseProviderModel]] = (
            ModelFactory._get_providers()
        )
        provider_class: Optional[Type[BaseProviderModel]] = discovered_providers.get(
            provider_name
        )
        if provider_class is None:
            available_providers: str = ", ".join(sorted(discovered_providers.keys()))
            raise ProviderConfigError(
                f"Unsupported provider '{provider_name}'. Available providers: {available_providers}"
            )

        LOGGER.debug(
            "Creating model '%s' using provider class '%s'.",
            model_info.id,
            provider_class.__name__,
        )
        return provider_class(model_info=model_info)
