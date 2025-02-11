import threading
import logging
from typing import Dict, List, Optional

from ember.core.registry.model import ModelFactory
from ember.core.registry.model.schemas.model_info import ModelInfo
from ember.core.registry.model.providers.base_provider import BaseProviderModel
from ember.core.registry.model.utils.model_registry_exceptions import ModelNotFoundError

logger: logging.Logger = logging.getLogger(__name__)


class ModelRegistry:
    """Thread-safe registry for managing model instances and metadata.

    This registry is intended for explicit dependency injection. Rather than using global
    singleton instances, pass an instance (typically via EmberAppContext) to components that
    require registry functionality.

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
        self._models: Dict[str, BaseProviderModel] = {}
        self._model_infos: Dict[str, ModelInfo] = {}
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
            ValueError: If a model with the provided model ID is already registered.
        """
        with self._lock:
            if model_info.model_id in self._models:
                raise ValueError(
                    "Model '{}' is already registered.".format(model_info.model_id)
                )
            model: BaseProviderModel = ModelFactory.create_model_from_info(
                model_info=model_info
            )
            self._models[model_info.model_id] = model
            self._model_infos[model_info.model_id] = model_info
            self._logger.info("Registered model: %s", model_info.model_id)

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
            self._models[model_info.model_id] = model
            self._model_infos[model_info.model_id] = model_info

    def get_model(self, model_id: str) -> BaseProviderModel:
        """Retrieves a registered model instance by its model ID.

        Args:
            model_id (str): Unique identifier of the model.

        Returns:
            BaseProviderModel: The model instance corresponding to the given model_id.

        Raises:
            ValueError: If the model_id is empty.
            ModelNotFoundError: If no model with the specified model_id is registered.
        """
        if not model_id:
            raise ValueError("Model ID cannot be empty")

        with self._lock:
            if model_id not in self._models:
                available_models: str = "\n- ".join(self._models.keys())
                raise ModelNotFoundError(
                    "Model '{}' not found. Available models:\n- {}".format(
                        model_id, available_models
                    )
                )
            try:
                return self._models[model_id]
            except KeyError as error:
                self._logger.error(
                    "Registry consistency error - missing model: %s", model_id
                )
                raise ModelNotFoundError(
                    "Model '{}' registration corrupted".format(model_id)
                ) from error

    def list_models(self) -> List[str]:
        """Lists all registered model IDs.

        Returns:
            List[str]: A list of model IDs currently registered.
        """
        with self._lock:
            return list(self._models.keys())

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
            if model_id in self._models:
                del self._models[model_id]
                self._logger.info("Unregistered model: %s", model_id)
            else:
                self._logger.warning(
                    "Model '%s' not found for unregistration.", model_id
                )
