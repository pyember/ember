"""Provides a factory for instantiating provider models from ModelInfo configurations.

This module facilitates dynamic discovery and instantiation of provider-specific model
instances based on validated model identifiers and discovered provider classes.
"""

import importlib
import inspect
import logging
import pkgutil
from typing import Any, Dict, Optional, Type

from src.ember.core.registry.model.base.utils.model_registry_exceptions import ProviderConfigError
from src.ember.core.registry.model.providers.base_provider import BaseProviderModel
from src.ember.core.registry.model.base.schemas.model_info import ModelInfo
from src.ember.core.registry.model.base.registry.model_enum import parse_model_str

logger: logging.Logger = logging.getLogger(__name__)


def discover_providers_in_package(
    *, package_path: str
) -> Dict[str, Type[BaseProviderModel]]:
    """Discovers provider classes within a specified package.

    This function dynamically traverses modules in the provided package,
    inspects each module for valid provider model classes (i.e. subclasses of
    BaseProviderModel excluding BaseProviderModel itself), and constructs a mapping from the
    provider's designated name (via the PROVIDER_NAME attribute) to the corresponding class.

    Args:
        package_path (str): The fully qualified package path in which provider modules reside.

    Returns:
        Dict[str, Type[BaseProviderModel]]: A dictionary mapping provider names to their classes.
    """
    providers: Dict[str, Type[BaseProviderModel]] = {}
    for _, module_name, _ in pkgutil.walk_packages(path=[package_path]):
        full_module_name: str = f"{package_path}.{module_name}"
        module: Any = importlib.import_module(full_module_name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, BaseProviderModel) and cls is not BaseProviderModel:
                provider_name: str = getattr(cls, "PROVIDER_NAME", "")
                if provider_name:
                    providers[provider_name] = cls
    return providers


class ModelFactory:
    """Factory for creating provider-specific model instances based on ModelInfo configurations.

    This factory validates the supplied model identifier, discovers available provider classes,
    and instantiates the corresponding provider model using the provided model configuration.
    """

    _provider_cache: Optional[Dict[str, Type[BaseProviderModel]]] = None

    @classmethod
    def _get_providers(cls) -> Dict[str, Type[BaseProviderModel]]:
        if cls._provider_cache is None:
            package_path: str = "ember.core.registry.model.providers"
            cls._provider_cache = discover_providers_in_package(
                package_path=package_path
            )
        return cls._provider_cache

    @staticmethod
    def create_model_from_info(*, model_info: ModelInfo) -> BaseProviderModel:
        """Instantiate and return a provider model using the supplied ModelInfo configuration.

        The method performs the following steps:
            1. Validates the model identifier using parse_model_str.
            2. Discovers all available provider classes in the predefined package path.
            3. Retrieves the provider class corresponding to model_info.provider.name.
            4. Instantiates the provider model using named parameter invocation.

        Args:
            model_info (ModelInfo): Configuration details for the model, including the id and provider.

        Returns:
            BaseProviderModel: An instantiated provider model corresponding to the provided configuration.

        Raises:
            ProviderConfigError: If the model identifier is invalid or the provider is unsupported.
        """
        # Validate the model ID.
        try:
            parse_model_str(model_info.id)
        except ValueError as value_error:
            raise ProviderConfigError(
                f"Unrecognized model ID '{model_info.id}'."
            ) from value_error

        provider_name: str = model_info.provider.name
        discovered_providers: Dict[str, Type[BaseProviderModel]] = (
            ModelFactory._get_providers()
        )

        provider_class: Optional[Type[BaseProviderModel]] = discovered_providers.get(
            provider_name
        )
        if provider_class is None:
            raise ProviderConfigError(f"Unsupported provider '{provider_name}'.")

        logger.debug(
            "Creating model '%s' using provider class '%s'.",
            model_info.id,
            provider_class.__name__,
        )
        model_instance: BaseProviderModel = provider_class(model_info=model_info)
        return model_instance
