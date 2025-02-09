import importlib
import inspect
import logging
import pkgutil
from typing import Dict, List, Optional, Type

from .utils.model_registry_exceptions import ProviderConfigError
from .provider_registry.base_provider import BaseProviderModel
from .core.schemas.model_info import ModelInfo
from .model_enum import parse_model_str

logger: logging.Logger = logging.getLogger(__name__)


def discover_providers(package: str) -> Dict[str, Type[BaseProviderModel]]:
    """Discover and return provider classes from the specified package.

    This function dynamically traverses modules in the provider registry package,
    inspects each module for classes that are valid provider models (i.e. subclasses
    of BaseProviderModel excluding the base class itself), and returns a dictionary
    mapping provider names (from the 'PROVIDER_NAME' attribute) to the corresponding
    provider class.

    Args:
        package (str): The fully-qualified package name where provider modules are located.

    Returns:
        Dict[str, Type[BaseProviderModel]]: Dictionary mapping provider names to their classes.
    """
    providers: Dict[str, Type[BaseProviderModel]] = {}
    package_paths: List[str] = ["src/ember/core/registry/model/provider_registry"]
    package_prefix: str = "src.ember.core.registry.model.provider_registry."

    for _, module_name, _ in pkgutil.walk_packages(
        path=package_paths, prefix=package_prefix
    ):
        module = importlib.import_module(module_name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, BaseProviderModel) and cls is not BaseProviderModel:
                provider_name: str = getattr(cls, "PROVIDER_NAME", "")
                if provider_name:
                    providers[provider_name] = cls

    return providers


class ModelFactory:
    """Factory class for instantiating provider models from ModelInfo configurations.

    This class provides a mechanism to create a provider-specific model instance
    using a validated model identifier and a dynamically discovered provider class.
    """

    @staticmethod
    def create_model_from_info(*, model_info: ModelInfo) -> BaseProviderModel:
        """Create and return an instance of a provider model based on the given ModelInfo.

        The method validates the model identifier, retrieves the appropriate provider
        class using dynamic discovery, and instantiates the provider model. If the
        model ID is unrecognized or the provider is unsupported, a ProviderConfigError
        is raised.

        Args:
            model_info (ModelInfo): The configuration object containing model details and provider info.

        Returns:
            BaseProviderModel: An instance of the provider model, initialized with the provided ModelInfo.

        Raises:
            ProviderConfigError: If the model ID cannot be parsed or the provider is unsupported.
        """
        try:
            validated_model_id: str = parse_model_str(model_info.model_id)
        except ValueError as exc:
            raise ProviderConfigError(
                f"Unrecognized model ID '{model_info.model_id}'."
            ) from exc

        provider_name: str = model_info.provider.name
        available_providers: Dict[str, Type[BaseProviderModel]] = discover_providers(
            package="src.ember.core.registry.model.provider_registry"
        )
        provider_class: Optional[Type[BaseProviderModel]] = available_providers.get(
            provider_name
        )
        if provider_class is None:
            raise ProviderConfigError(f"Unsupported provider '{provider_name}'.")

        logger.debug(
            "Creating model '%s' using provider class '%s'.",
            validated_model_id,
            provider_class.__name__,
        )
        model_instance: BaseProviderModel = provider_class(model_info=model_info)
        return model_instance
