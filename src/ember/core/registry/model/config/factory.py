import importlib
import inspect
import logging
import pkgutil
from typing import Dict, List, Optional, Type

from ember.core.registry.model.utils.model_registry_exceptions import (
    ProviderConfigError,
)
from ember.core.registry.model.provider_registry.base import BaseProviderModel
from ember.core.registry.model.core.schemas.model_info import ModelInfo
from ember.core.registry.model.config.model_enum import parse_model_str

logger: logging.Logger = logging.getLogger(__name__)


def discover_providers(package: str) -> Dict[str, Type[BaseProviderModel]]:
    """Recursively discover provider classes within the specified package.

    This function traverses the provided package and its subpackages, imports
    modules, and collects classes that subclass BaseProviderModel and specify a
    non-empty PROVIDER_NAME attribute.

    Args:
        package (str): Dot-separated package name to inspect (e.g.,
            "src.ember.registry.model.provider_registry").

    Returns:
        Dict[str, Type[BaseProviderModel]]: A mapping from provider names to their
            corresponding provider class.
    """
    provider_map: Dict[str, Type[BaseProviderModel]] = {}
    package_path: List[str] = [package.replace(".", "/")]
    package_prefix: str = f"{package}."

    for _, module_name, _ in pkgutil.walk_packages(
        path=package_path, prefix=package_prefix
    ):
        module = importlib.import_module(module_name)
        for _, member_class in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(member_class, BaseProviderModel)
                and member_class is not BaseProviderModel
            ):
                provider_name: str = getattr(member_class, "PROVIDER_NAME", "")
                if provider_name:
                    provider_map[provider_name] = member_class

    return provider_map


class ModelFactory:
    """Factory for creating provider model instances based on model metadata."""

    @staticmethod
    def create_model_from_info(model_info: ModelInfo) -> BaseProviderModel:
        """Creates a provider model instance from the provided model metadata.

        This method validates the model identifier, discovers available provider
        classes, and instantiates the specific provider model. It raises an error if
        the model identifier is unrecognized or if the provider is unsupported.

        Args:
            model_info (ModelInfo): Model metadata required for instantiation.

        Raises:
            ProviderConfigError: If the model ID is unrecognized or the provider is unsupported.

        Returns:
            BaseProviderModel: An instance of the provider model.
        """
        try:
            validated_model_id: str = parse_model_str(model_str=model_info.model_id)
        except ValueError as error:
            raise ProviderConfigError(
                f"Unrecognized model ID '{model_info.model_id}'."
            ) from error

        provider_name: str = model_info.provider.name
        providers_map: Dict[str, Type[BaseProviderModel]] = discover_providers(
            package="src.ember.registry.model.provider_registry"
        )
        model_class: Optional[Type[BaseProviderModel]] = providers_map.get(
            provider_name
        )
        if model_class is None:
            raise ProviderConfigError(f"Unsupported provider '{provider_name}'.")

        logger.debug(
            "Creating model '%s' using provider class '%s'.",
            validated_model_id,
            model_class.__name__,
        )
        return model_class(model_info=model_info)
