import threading
import logging
from typing import Dict, List, Optional, Generic, TypeVar

from ember.core.registry.model.base.registry.factory import ModelFactory
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.providers.base_provider import BaseProviderModel
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    ModelNotFoundError,
)

# Type variable for model implementation
M = TypeVar("M", bound=BaseProviderModel)

logger: logging.Logger = logging.getLogger(__name__)


class ModelRegistry(Generic[M]):
    """Thread-safe registry for managing LLM provider model instances and their metadata.

    The ModelRegistry is a central component in the Ember framework that manages the 
    lifecycle of language model instances. It provides a unified interface for registering, 
    retrieving, and managing different language models from various providers.
    
    Key features:
    - Thread-safe operations for concurrent access
    - Lazy instantiation of model instances to minimize resource usage
    - Generic typing to support different model implementations
    - Centralized model metadata management
    - Model lifecycle management (registration, retrieval, unregistration)
    
    Threading model:
    All public methods of this class are thread-safe, protected by an internal lock.
    This allows multiple threads to interact with the registry concurrently without
    data corruption or race conditions.
    
    Lazy instantiation:
    Models are only instantiated when first requested via get_model(), not at registration time.
    This improves performance and resource usage for applications that register many models
    but only use a subset of them.
    
    Usage example:
    ```python
    # Create a registry
    registry = ModelRegistry()
    
    # Register a model
    model_info = ModelInfo(
        id="openai:gpt-4",
        provider=ProviderInfo(name="openai", default_api_key="YOUR_API_KEY")
    )
    registry.register_model(model_info)
    
    # Get and use a model
    model = registry.get_model("openai:gpt-4")
    response = model("Hello, world!")
    print(response.data)
    ```

    Type Parameters:
        M: The type of models stored in this registry (defaults to BaseProviderModel)

    Attributes:
        _lock (threading.Lock): A lock ensuring thread-safe operations.
        _models (Dict[str, M]): Mapping from model IDs to model instances.
        _model_infos (Dict[str, ModelInfo]): Mapping from model IDs to their metadata.
        _logger (logging.Logger): Logger instance specific to ModelRegistry.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initializes a new instance of ModelRegistry.

        Args:
            logger (Optional[logging.Logger]): Optional logger to use. If None, a default logger is created.
        """
        self._lock: threading.Lock = threading.Lock()
        self._model_infos: Dict[str, ModelInfo] = {}
        self._models: Dict[str, M] = {}
        self._logger: logging.Logger = logger or logging.getLogger(
            self.__class__.__name__
        )

    def register_model(self, model_info: ModelInfo) -> None:
        """
        Registers a new model using its metadata.

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
        """
        Registers a new model or updates an existing model with provided metadata.

        This method uses the ModelFactory to instantiate (or re-instantiate) the model and updates the registry
        with the latest model instance and its associated metadata.

        Args:
            model_info (ModelInfo): The configuration and metadata for model instantiation or update.
        """
        with self._lock:
            model = ModelFactory.create_model_from_info(model_info=model_info)
            self._models[model_info.id] = model
            self._model_infos[model_info.id] = model_info

    def get_model(self, model_id: str) -> M:
        """
        Lazily instantiate the model when first requested.

        Args:
            model_id: Unique identifier of the model

        Returns:
            The model instance of type M

        Raises:
            ValueError: If model_id is empty
            ModelNotFoundError: If the model is not registered
        """
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
        """
        Check if a model is registered without instantiating it.

        Args:
            model_id: Unique identifier of the model.

        Returns:
            True if the model is registered, False otherwise.
        """
        with self._lock:
            return model_id in self._model_infos

    def list_models(self) -> List[str]:
        """
        Lists all registered model IDs.

        Returns:
            A list of *registered* model IDs (lazy loaded or not).
        """
        with self._lock:
            return list(self._model_infos.keys())

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Retrieves metadata for a registered model by its model ID.

        Args:
            model_id: Unique identifier of the model.

        Returns:
            The model's metadata if registered; otherwise, None.
        """
        with self._lock:
            return self._model_infos.get(model_id)

    def unregister_model(self, model_id: str) -> None:
        """
        Unregisters a model by its model ID.

        Args:
            model_id: Unique identifier of the model to unregister.
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
