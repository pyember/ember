import threading
from typing import Dict, List, Optional

from .core.schemas.model_info import ModelInfo
from .factory import ModelFactory
from .provider_registry.base import BaseProviderModel


class ModelRegistry:
    """Thread-safe registry for managing model instances and their associated metadata.

    This registry provides a thread-safe mechanism to register, update, and retrieve model
    instances along with their metadata. For everyday usage, higher-level abstractions (such as
    ModelService or LMModule) are recommended. Direct access is reserved for advanced orchestration.

    Attributes:
        _lock (threading.Lock): A lock object ensuring thread-safe operations.
        _models (Dict[str, BaseProviderModel]): A mapping from model IDs to model instances.
        _model_infos (Dict[str, ModelInfo]): A mapping from model IDs to their metadata.
    """

    def __init__(self) -> None:
        """Initializes a new instance of ModelRegistry."""
        self._lock: threading.Lock = threading.Lock()
        self._models: Dict[str, BaseProviderModel] = {}
        self._model_infos: Dict[str, ModelInfo] = {}

    def register_model(self, model_info: ModelInfo) -> None:
        """Registers a new model using its metadata.

        This method creates a model instance using the ModelFactory and registers it along with
        its metadata. It raises a ValueError if a model with the same ID is already registered.

        Args:
            model_info (ModelInfo): The configuration and metadata necessary to create the model.

        Raises:
            ValueError: If a model with the provided model ID is already registered.
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
        """Registers a new model or updates an existing model using the provided metadata.

        This method leverages the ModelFactory to create (or recreate) a model instance and updates
        the registry with the latest model instance and metadata.

        Args:
            model_info (ModelInfo): The configuration and metadata used to instantiate or update the model.
        """
        with self._lock:
            model: BaseProviderModel = ModelFactory.create_model_from_info(
                model_info=model_info
            )
            self._models[model_info.model_id] = model
            self._model_infos[model_info.model_id] = model_info

    def get_model(self, model_id: str) -> Optional[BaseProviderModel]:
        """Retrieves a registered model instance by its model ID.

        Args:
            model_id (str): The unique identifier of the model.

        Returns:
            Optional[BaseProviderModel]: The model instance if found; otherwise, None.
        """
        with self._lock:
            return self._models.get(model_id)

    def list_models(self) -> List[str]:
        """Lists all registered model IDs.

        Returns:
            List[str]: A list of all model IDs currently registered.
        """
        with self._lock:
            return list(self._models.keys())

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Retrieves metadata for a registered model by its model ID.

        Args:
            model_id (str): The unique identifier of the model.

        Returns:
            Optional[ModelInfo]: The model's metadata if available; otherwise, None.
        """
        with self._lock:
            return self._model_infos.get(model_id)


GLOBAL_MODEL_REGISTRY: ModelRegistry = ModelRegistry()
