import logging
import importlib
import inspect
import pkgutil
from typing import Dict, Type

from src.avior.registry.model.exceptions import ProviderConfigError
from src.avior.registry.model.provider_registry.base import BaseProviderModel
from src.avior.registry.model.schemas.model_info import ModelInfo
from src.avior.registry.model.schemas.provider_info import ProviderInfo
from src.avior.registry.model.schemas.cost import ModelCost, RateLimit
from src.avior.registry.model.registry.model_enum import parse_model_str

logger = logging.getLogger(__name__)

def discover_providers(package: str) -> Dict[str, Type[BaseProviderModel]]:
    """
    Recursively walks through 'package' and all subpackages,
    looking for classes that inherit from BaseProviderModel
    and have a PROVIDER_NAME attribute.
    """
    provider_map = {}

    # pkgutil.walk_packages takes an iterable of paths and an optional prefix,
    # letting us discover nested modules/subpackages.
    for finder, mod_name, is_pkg in pkgutil.walk_packages(
        path=[package.replace(".", "/")],
        prefix=package + ".",
    ):
        # Import the module
        module = importlib.import_module(mod_name)
        # Inspect all classes, see if they belong to BaseProviderModel
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseProviderModel) and obj is not BaseProviderModel:
                provider_name = getattr(obj, "PROVIDER_NAME", None)
                if provider_name:
                    provider_map[provider_name] = obj

    return provider_map

class ModelFactory:
    """Factory that instantiates provider model classes from ModelInfo."""
    @staticmethod
    def create_model_from_info(model_info: ModelInfo) -> BaseProviderModel:
        try:
            validated_id = parse_model_str(model_info.model_id)
        except ValueError:
            raise ProviderConfigError(f"Unrecognized model ID '{model_info.model_id}'.")

        provider_name = model_info.provider.name
        discovered_map = discover_providers("src.avior.registry.model.provider_registry")  # now recursive
        model_class = discovered_map.get(provider_name)
        if not model_class:
            raise ProviderConfigError(f"Unsupported provider '{provider_name}'.")

        logger.debug("Creating model '%s' via %s", validated_id, model_class.__name__)
        return model_class(model_info)
