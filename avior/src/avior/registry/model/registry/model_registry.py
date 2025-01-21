import threading
from typing import Dict, Optional

from src.avior.registry.model.schemas.model_info import ModelInfo
from src.avior.registry.model.registry.factory import ModelFactory
from src.avior.registry.model.provider_registry.base import BaseProviderModel


GLOBAL_MODEL_REGISTRY = None  # Will be initialized at import time

class ModelRegistry:
    """
    Thread-safe registry for managing multiple model instances.

    Responsibilities:
    - Create and store model instances from ModelInfo objects.
    - Provide concurrency-safe get/set operations.
    - Return None if a requested model is not found (the caller can raise an error).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._models: Dict[str, BaseProviderModel] = {}
        self._model_infos: Dict[str, ModelInfo] = {}

    def register_model(self, model_info: ModelInfo) -> None:
        """
        Create the provider instance from `model_info` and store it in the registry.
        Raises ValueError if model_id is already registered.

        :param model_info: A ModelInfo object describing how to create and configure a model.
        """
        with self._lock:
            if model_info.model_id in self._models:
                raise ValueError(
                    f"Model '{model_info.model_id}' is already registered."
                )
            model = ModelFactory.create_model_from_info(model_info)
            self._models[model_info.model_id] = model
            self._model_infos[model_info.model_id] = model_info

    def get_model(self, model_id: str) -> Optional[BaseProviderModel]:
        """
        Retrieves the model instance if it exists. Returns None otherwise.

        :param model_id: The string ID of the model as used in the registry.
        :return: The BaseProviderModel instance or None.
        """
        with self._lock:
            return self._models.get(model_id)

    def list_models(self) -> list[str]:
        """
        Returns a list of registered model IDs for debugging or introspection.
        """
        with self._lock:
            return list(self._models.keys())

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Returns the ModelInfo associated with the given model_id, or None if not found.

        :param model_id: The string ID of the model.
        :return: The ModelInfo object or None.
        """
        with self._lock:
            return self._model_infos.get(model_id)


# Instantiate a global ModelRegistry instance at import time
# so there's exactly one shared registry in this module.
GLOBAL_MODEL_REGISTRY = ModelRegistry()
