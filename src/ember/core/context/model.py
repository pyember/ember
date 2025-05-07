"""Model management component.

This module provides a component for managing model instances and
their discovery and creation with thread-safe lazy initialization.
"""

import importlib
import logging
from typing import Any, Dict, List, Optional, Type

from .component import Component
from .registry import Registry


class ModelComponent(Component):
    """Model management component.

    This component is responsible for model discovery, registration,
    and retrieval with efficient caching and thread safety.

    Features:
    - Lazy loading: Models are only initialized when first accessed
    - Multiple registration methods: Direct instances or classes
    - Thread safety: Safe for concurrent access
    - Configuration-based: Models can be defined in configuration
    """

    def __init__(self, registry: Optional[Registry] = None):
        """Initialize with registry.

        Args:
            registry: Registry to use (current thread's if None)
        """
        super().__init__(registry)
        self._models: Dict[str, Any] = {}
        self._model_classes: Dict[str, Type] = {}
        self._logger = logging.getLogger("ember.model")

    def _register(self) -> None:
        """Register in registry as 'model'."""
        self._registry.register("model", self)

    def get_model(self, model_id: str) -> Optional[Any]:
        """Get model by ID.

        This method follows a three-tiered lookup strategy:
        1. Look in the models cache
        2. Try to instantiate from registered model classes
        3. Try to create from configuration

        Args:
            model_id: Model identifier

        Returns:
            Model instance or None if not found
        """
        self._ensure_initialized()

        # Return cached model if available
        if model_id in self._models:
            return self._models[model_id]

        # Try to create from registered class
        if model_id in self._model_classes:
            with self._lock:
                # Double-check after acquiring lock
                if model_id not in self._models:
                    try:
                        model_class = self._model_classes[model_id]
                        model = model_class()
                        self._models[model_id] = model
                    except Exception as e:
                        self._logger.error(f"Error creating model '{model_id}': {e}")
                        return None
            return self._models.get(model_id)

        # Try to load from configuration
        config = self._registry.get("config")
        if config:
            model_config = config.get_config("models").get(model_id, {})
            if model_config:
                with self._lock:
                    # Double-check after acquiring lock
                    if model_id not in self._models:
                        model = self._create_model_from_config(model_id, model_config)
                        if model:
                            self._models[model_id] = model
                return self._models.get(model_id)

        return None

    def register_model(self, model_id: str, model: Any) -> None:
        """Register a model instance.

        Args:
            model_id: Model identifier
            model: Model instance
        """
        self._ensure_initialized()
        with self._lock:
            self._models[model_id] = model

    def register_model_class(self, model_id: str, model_class: Type) -> None:
        """Register a model class for lazy instantiation.

        Args:
            model_id: Model identifier
            model_class: Model class to instantiate
        """
        self._ensure_initialized()
        with self._lock:
            self._model_classes[model_id] = model_class

    def list_models(self) -> List[str]:
        """List available model IDs.

        Returns:
            List of registered model IDs
        """
        self._ensure_initialized()
        return sorted(
            list(set(list(self._models.keys()) + list(self._model_classes.keys())))
        )

    def _initialize(self) -> None:
        """Initialize models from configuration.

        This loads model definitions from configuration and
        prepares them for lazy instantiation.
        """
        config = self._registry.get("config")
        if not config:
            return

        model_configs = config.get_config("models")
        if not model_configs:
            return

        for model_id, model_config in model_configs.items():
            # Skip models that are already registered
            if model_id in self._models or model_id in self._model_classes:
                continue

            # Register model class for lazy instantiation if provider is specified
            provider = model_config.get("provider")
            if provider:
                module_path = model_config.get("module")
                if module_path:
                    try:
                        self._register_model_from_module(
                            model_id, module_path, model_config
                        )
                    except Exception as e:
                        self._logger.error(f"Error registering model '{model_id}': {e}")

    def _register_model_from_module(
        self, model_id: str, module_path: str, config: Dict[str, Any]
    ) -> None:
        """Register model from module path.

        Args:
            model_id: Model identifier
            module_path: Python module path
            config: Model configuration
        """
        try:
            module_parts = module_path.split(".")
            class_name = module_parts[-1]
            module_path = ".".join(module_parts[:-1])

            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            # Create a factory function that passes config
            def model_factory():
                return model_class(config)

            # Register factory
            self._model_classes[model_id] = model_factory
        except (ImportError, AttributeError) as e:
            self._logger.error(f"Failed to load model from {module_path}: {e}")

    def _create_model_from_config(
        self, model_id: str, model_config: Dict[str, Any]
    ) -> Optional[Any]:
        """Create a model from configuration.

        Args:
            model_id: Model identifier
            model_config: Model configuration

        Returns:
            Model instance or None if creation failed
        """
        provider = model_config.get("provider")
        if not provider:
            self._logger.error(f"No provider specified for model {model_id}")
            return None

        try:
            # Import provider module
            provider_module = importlib.import_module(f"ember.core.models.{provider}")
            create_func = provider_module.create_model

            # Create model
            model = create_func(model_config)
            return model
        except (ImportError, AttributeError) as e:
            self._logger.error(f"Failed to create model {model_id}: {e}")
            return None
