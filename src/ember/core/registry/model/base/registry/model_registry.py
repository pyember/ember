import logging
import threading
from typing import Dict, Generic, List, Optional, TypeVar

from ember.core.registry.model.base.registry.factory import ModelFactory
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    ModelNotFoundError,
)
from ember.core.registry.model.providers.base_provider import BaseProviderModel

# Type variable for model implementation
M = TypeVar("M", bound=BaseProviderModel)

logger: logging.Logger = logging.getLogger(__name__)
# Set default log level to WARNING to reduce verbosity
logger.setLevel(logging.WARNING)


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

    # Register a model (simple version with minimal required fields)
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
        # Fast path for empty model_id
        if not model_id:
            return False

        # Use the lock non-blockingly if possible to optimize for the common case
        # where we don't already hold the lock
        if self._lock.acquire(blocking=False):
            try:
                return model_id in self._model_infos
            finally:
                self._lock.release()
        else:
            # We're likely already holding the lock in the current thread
            # so just check directly to avoid deadlock
            return model_id in self._model_infos

    def list_models(self) -> List[str]:
        """Lists all registered model IDs.

        Returns:
            List[str]: A list of registered model IDs (lazy loaded or not).
        """
        with self._lock:
            return list(self._model_infos.keys())

    def discover_models(self) -> List[str]:
        """Discovers models using the ModelDiscoveryService and registers them.

        This method encapsulates the model discovery process by:
        1. Creating a ModelDiscoveryService instance
        2. Retrieving model metadata from provider APIs
        3. Merging with local configuration
        4. Registering newly discovered models

        The discovery process is resilient to failures, with detailed logging
        of each step and proper error handling.

        Returns:
            List[str]: A list of newly discovered and registered model IDs.

        Note:
            This method is thread-safe and can be called concurrently from
            multiple threads. Any failures during discovery are logged but
            will not raise exceptions to the caller.
        """
        # Import here to avoid circular dependency issues
        from ember.core.exceptions import EmberError
        from ember.core.registry.model.base.registry.discovery import (
            ModelDiscoveryService,
        )

        discovery_service = ModelDiscoveryService()
        self._logger.info("Initiating model discovery via ModelDiscoveryService")

        newly_registered: List[str] = []

        try:
            # Step 1: Discover raw model metadata from provider APIs
            discovered_models = discovery_service.discover_models()
            if not discovered_models:
                self._logger.info("No models discovered from provider APIs")
                return newly_registered

            self._logger.debug(
                "Raw discovery found %d models: %s",
                len(discovered_models),
                list(discovered_models.keys()),
            )

            # Step 2: Merge discovered models with local configuration
            merged_models = discovery_service.merge_with_config(
                discovered=discovered_models
            )
            if not merged_models:
                self._logger.info(
                    "No models remaining after merging with configuration"
                )
                return newly_registered

            self._logger.debug(
                "Merged discovery found %d models: %s",
                len(merged_models),
                list(merged_models.keys()),
            )

            # Step 3: Register each model, tracking newly added ones
            self._logger.info(f"Registering {len(merged_models)} models from discovery")
            registration_stats = {"new": 0, "skipped": 0, "failed": 0}

            # Process models one by one with appropriate locking
            for model_id, model_info in merged_models.items():
                # Check if registered with minimal lock scope
                is_already_registered = False
                with self._lock:
                    is_already_registered = model_id in self._model_infos

                if not is_already_registered:
                    try:
                        self._logger.debug(
                            "Attempting to register discovered model: %s (provider: %s)",
                            model_id,
                            (
                                model_info.provider.name
                                if model_info.provider
                                else "unknown"
                            ),
                        )
                        # Use register_model which has its own lock
                        self.register_model(model_info=model_info)
                        newly_registered.append(model_id)
                        registration_stats["new"] += 1
                        self._logger.info(
                            "Successfully registered model: %s with provider %s",
                            model_id,
                            (
                                model_info.provider.name
                                if model_info.provider
                                else "unknown"
                            ),
                        )
                    except ValueError as registration_error:
                        # Expected error if model already exists
                        registration_stats["skipped"] += 1
                        self._logger.debug(
                            "Model %s already registered: %s",
                            model_id,
                            registration_error,
                        )
                    except Exception as unexpected_error:
                        # Unexpected error during registration (validation failure, etc.)
                        registration_stats["failed"] += 1
                        self._logger.error(
                            "Failed to register discovered model %s: %s (error type: %s)",
                            model_id,
                            unexpected_error,
                            type(unexpected_error).__name__,
                        )

            self._logger.info(
                "Registration summary: %d new, %d skipped, %d failed",
                registration_stats["new"],
                registration_stats["skipped"],
                registration_stats["failed"],
            )

            if newly_registered:
                self._logger.info(
                    "Successfully discovered and registered %d new models: %s",
                    len(newly_registered),
                    newly_registered,
                )
            else:
                self._logger.info(
                    "No new models discovered that weren't already registered"
                )

            return newly_registered

        except EmberError as ember_error:
            self._logger.error(
                "Model discovery failed with framework error: %s", ember_error
            )
            return newly_registered
        except Exception as unexpected_error:
            self._logger.exception(
                "Unexpected error during model discovery: %s", unexpected_error
            )
            return newly_registered

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
