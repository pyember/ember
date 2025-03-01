import threading
import logging
from typing import Dict, List, Optional

from ember.core.registry.model.base.registry.factory import ModelFactory
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.providers.base_provider import BaseProviderModel
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    ModelNotFoundError,
)

logger: logging.Logger = logging.getLogger(__name__)


class ModelRegistry:
    """Thread-safe registry for managing model instances and metadata.

    This registry stores ModelInfo structures and lazily instantiates provider
    model objects. It also handles basic thread safety and concurrency.

    Attributes:
        _lock (threading.Lock): A lock ensuring thread-safe operations.
        _models (Dict[str, BaseProviderModel]): Mapping from model IDs to model instances.
        _model_infos (Dict[str, ModelInfo]): Mapping from model IDs to their metadata.
        _logger (logging.Logger): Logger instance specific to ModelRegistry.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initializes a new instance of ModelRegistry.

        Args:
            logger (Optional[logging.Logger]): Optional logger to use. If None, a default logger is created.
        """
        self._lock: threading.Lock = threading.Lock()
        self._model_infos: Dict[str, ModelInfo] = {}
        self._models: Dict[str, BaseProviderModel] = {}
        self._logger: logging.Logger = logger or logging.getLogger(
            self.__class__.__name__
        )

    def register_model(self, model_info: ModelInfo) -> None:
        """Registers a new model using its metadata.

        This method instantiates a model via the ModelFactory and registers it along with its metadata.
        A ValueError is raised if a model with the same ID is already registered.

        Args:
            model_info (ModelInfo): The configuration and metadata required to create the model.

        Raises:
            ValueError: If a model with the same ID is already registered.
        """
        with self._lock:
            if model_info.id in self._model_infos:
                raise ValueError(f"Model '{model_info.id}' is already registered.")
            self._model_infos[model_info.id] = model_info
            self._logger.info(
                "Successfully registered model: %s with provider %s",
                model_info.id,
                model_info.provider.name,
            )

    def register_or_update_model(self, model_info: ModelInfo) -> None:
        """Registers a new model or updates an existing model with provided metadata.

        This method uses the ModelFactory to instantiate (or re-instantiate) the model and updates the registry
        with the latest model instance and its associated metadata.

        Args:
            model_info (ModelInfo): The configuration and metadata for model instantiation or update.
        """
        with self._lock:
            model: BaseProviderModel = ModelFactory.create_model_from_info(
                model_info=model_info
            )
            self._models[model_info.id] = model
            self._model_infos[model_info.id] = model_info

    def get_model(self, model_id: str) -> BaseProviderModel:
        """Lazily instantiate the model when first requested."""
        if not model_id:
            raise ValueError("Model ID cannot be empty")

        with self._lock:
            if model_id not in self._model_infos:
                available_models: str = "\n- ".join(self._model_infos.keys())
                raise ModelNotFoundError(
                    f"Model '{model_id}' not found. Available models:\n- {available_models}"
                )
            if model_id not in self._models:
                model_info = self._model_infos[model_id]
                model = ModelFactory.create_model_from_info(model_info=model_info)
                self._models[model_id] = model
                self._logger.info("Instantiated model: %s", model_id)
            return self._models[model_id]

    def is_registered(self, model_id: str) -> bool:
        """Check if a model is registered without instantiating it.

        Args:
            model_id (str): Unique identifier of the model.

        Returns:
            bool: True if the model is registered, False otherwise.
        """
        with self._lock:
            return model_id in self._model_infos

    def list_models(self) -> List[str]:
        """Lists all registered model IDs.

        Returns:
            List[str]: A list of *registered* model IDs (lazy loaded or not).
        """
        with self._lock:
            return list(self._model_infos.keys())

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Retrieves metadata for a registered model by its model ID.

        Args:
            model_id (str): Unique identifier of the model.

        Returns:
            Optional[ModelInfo]: The model's metadata if registered; otherwise, None.
        """
        with self._lock:
            return self._model_infos.get(model_id)

    def unregister_model(self, model_id: str) -> None:
        """Unregisters a model by its model ID.

        Args:
            model_id (str): Unique identifier of the model to unregister.
        """
        with self._lock:
            if model_id in self._model_infos:
                del self._model_infos[model_id]
                if model_id in self._models:
                    del self._models[model_id]
                self._logger.info("Successfully unregistered model: %s", model_id)
            else:
                self._logger.warning(
                    "Attempted to unregister non-existent model '%s'.", model_id
                )
