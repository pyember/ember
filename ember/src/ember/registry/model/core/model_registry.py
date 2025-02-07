import threading
from typing import Dict, Optional, List

from src.ember.registry.model.schemas.model_info import ModelInfo
from src.ember.registry.model.registry.factory import ModelFactory
from src.ember.registry.model.provider_registry.base import BaseProviderModel

# Global ModelRegistry instance instantiated at import time.
GLOBAL_MODEL_REGISTRY: "ModelRegistry"  # Type hint for a global instance.


class ModelRegistry:
    """Thread-safe registry for managing model instances.

    This registry is responsible for creating, registering, and retrieving
    model instances constructed from ModelInfo objects. All operations are
    concurrency-safe.
    """

    def __init__(self) -> None:
        """Initializes a new ModelRegistry instance.

        This method sets up the necessary locks and internal dictionaries
        for storing both provider model instances and their metadata.
        """
        self._lock: threading.Lock = threading.Lock()
        self._models: Dict[str, BaseProviderModel] = {}
        self._model_infos: Dict[str, ModelInfo] = {}

    def register_model(self, model_info: ModelInfo) -> None:
        """Registers a new model in the registry.

        Creates a provider instance based on the supplied model_info and adds
        it to the registry. Raises a ValueError if a model with the same ID
        is already registered.

        Args:
            model_info (ModelInfo): The model information used to create and configure the model.

        Raises:
            ValueError: If a model with the same model_id already exists in the registry.
        """
        with self._lock:
            if model_info.model_id in self._models:
                raise ValueError(
                    f"Model '{model_info.model_id}' is already registered."
                )
            model: BaseProviderModel = ModelFactory.create_model_from_info(
                model_info=model_info
            )
            self._models[model_info.model_id] = model
            self._model_infos[model_info.model_id] = model_info

    def register_or_update_model(self, model_info: ModelInfo) -> None:
        """Registers a new model or updates an existing model's metadata.

        If the model is already registered, its provider instance and metadata are updated.
        Otherwise, the model is newly registered with the given model_info.

        Args:
            model_info (ModelInfo): The model information for registration or update.
        """
        with self._lock:
            model: BaseProviderModel = ModelFactory.create_model_from_info(
                model_info=model_info
            )
            self._models[model_info.model_id] = model
            self._model_infos[model_info.model_id] = model_info

    def get_model(self, model_id: str) -> Optional[BaseProviderModel]:
        """Retrieves a registered model instance by its identifier.

        Args:
            model_id (str): The unique identifier of the model.

        Returns:
            Optional[BaseProviderModel]: The model instance if found, or None otherwise.
        """
        with self._lock:
            return self._models.get(model_id)

    def list_models(self) -> List[str]:
        """Lists all registered model identifiers.

        Returns:
            List[str]: A list of all model IDs currently stored in the registry.
        """
        with self._lock:
            return list(self._models.keys())

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Retrieves the metadata associated with a given model identifier.

        Args:
            model_id (str): The unique identifier of the model.

        Returns:
            Optional[ModelInfo]: The corresponding ModelInfo if available, or None.
        """
        with self._lock:
            return self._model_infos.get(model_id)


# Instantiates a global ModelRegistry instance at import time.
# This ensures a single shared registry within this module.
GLOBAL_MODEL_REGISTRY = ModelRegistry()
