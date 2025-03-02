import logging
import threading
from threading import Lock
from typing import Optional, Dict, ClassVar, Type, Any

from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.base.services.usage_service import UsageService
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.types.config_types import ConfigManager


class EmberAppContext:
    """
    Application context for Ember, holding references to core services.

    This context serves as the composition root for dependency injection, eliminating
    reliance on global singleton state. For architectural details, see ARCHITECTURE.md.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        model_registry: ModelRegistry,
        usage_service: Optional[UsageService],
        logger: logging.Logger,
    ) -> None:
        """
        Initialize the application context with core services.

        Args:
            config_manager: Configuration manager implementation
            model_registry: Model registry service
            usage_service: Optional usage tracking service
            logger: Logger instance
        """
        self.config_manager = config_manager
        self.model_registry = model_registry
        self.usage_service = usage_service
        self.logger = logger


def create_ember_app(config_filename: Optional[str] = None) -> EmberAppContext:
    """
    Creates and returns an initialized EmberAppContext.

    This composition root:
      1. Creates a logger.
      2. Instantiates the ConfigManager.
      3. Injects API keys using a unified logging approach.
      4. Creates the ModelRegistry and auto-registers known models.
      5. Instantiates additional services.

    Args:
        config_filename: Optional path to config file

    Returns:
        Initialized EmberAppContext with all services
    """
    logger = logging.getLogger("ember")

    from ember.core.configs.config import (
        ConfigManager,
        initialize_api_keys,
        auto_register_known_models,
    )

    # 1) Create the configuration manager with dependency injection.
    config_manager = ConfigManager(config_filename=config_filename, logger=logger)

    # 2) Initialize API keys, passing in the consistent logger.
    initialize_api_keys(config_manager, logger=logger)

    # 3) Create the model registry and auto-register models.
    model_registry = ModelRegistry(logger=logger)
    auto_register_known_models(registry=model_registry, config_manager=config_manager)

    # 4) Create additional services (e.g., usage service).
    usage_service = UsageService(logger=logger)

    # 5) Return the aggregated application context.
    return EmberAppContext(
        config_manager=config_manager,
        model_registry=model_registry,
        usage_service=usage_service,
        logger=logger,
    )


class EmberContext:
    """
    Global application context for Ember.

    This class implements the Singleton pattern to provide global access to
    the EmberAppContext instance, which contains all core services.

    The EmberContext lazy-loads the app context on first access, avoiding
    circular imports and allowing more flexible initialization patterns.
    
    For testing purposes, a test mode can be enabled that bypasses the singleton
    pattern to allow isolated test contexts.
    """

    _instance: ClassVar[Optional["EmberContext"]] = None
    _lock: ClassVar[Lock] = Lock()
    _app_context: Optional[EmberAppContext] = None
    _test_mode: ClassVar[bool] = False
    _thread_local: ClassVar = threading.local()

    # Backwards compatibility attributes for transition
    model_registry: ModelRegistry

    @classmethod
    def enable_test_mode(cls) -> None:
        """
        Enable test mode to bypass singleton restrictions.
        
        In test mode, each call to EmberContext() creates a new instance,
        preventing deadlocks and allowing isolated testing.
        """
        cls._test_mode = True
        # Clear instance to ensure fresh instances even after a previous singleton was created
        cls._instance = None
        if hasattr(cls._thread_local, 'instances'):
            delattr(cls._thread_local, 'instances')
    
    @classmethod
    def disable_test_mode(cls) -> None:
        """
        Disable test mode and return to normal singleton behavior.
        """
        cls._test_mode = False
        cls._instance = None
        if hasattr(cls._thread_local, 'instances'):
            delattr(cls._thread_local, 'instances')

    def __getattr__(self, name: str) -> Any:
        """
        Fallback attribute access that delegates to the app_context.

        This enables backwards compatibility for code written against
        the old global registry pattern.

        Args:
            name: Attribute name

        Returns:
            The requested attribute from the app_context

        Raises:
            AttributeError: If the attribute doesn't exist on app_context
        """
        if self._app_context is None:
            self._app_context = create_ember_app()

        if hasattr(self._app_context, name):
            return getattr(self._app_context, name)

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __new__(cls) -> "EmberContext":
        if cls._test_mode:
            # In test mode, always create a new instance
            instance = super().__new__(cls)
            instance._app_context = None
            return instance
            
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._app_context = None
            return cls._instance

    @classmethod
    def initialize(
        cls,
        config_path: Optional[str] = None,
        app_context: Optional[EmberAppContext] = None,
    ) -> "EmberContext":
        """
        Initialize the global context with provided or default components.

        Args:
            config_path: Optional path to config file
            app_context: Optional pre-configured EmberAppContext

        Returns:
            The initialized EmberContext instance (singleton in normal mode, new in test mode)
        """
        if cls._test_mode:
            # In test mode, create a new instance
            instance = cls()
            
            # Set app_context or create a new one
            if app_context is not None:
                instance._app_context = app_context
            else:
                instance._app_context = create_ember_app(config_filename=config_path)
                
            return instance
            
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()

            # If app_context is provided, use it directly
            if app_context is not None:
                cls._instance._app_context = app_context
            # Otherwise create a new one if needed
            elif cls._instance._app_context is None:
                cls._instance._app_context = create_ember_app(
                    config_filename=config_path
                )

            return cls._instance

    @classmethod
    def get(cls) -> "EmberContext":
        """
        Get the current global context, initializing if needed.

        Returns:
            The EmberContext instance (singleton in normal mode, new in test mode)
        """
        if cls._test_mode:
            # In test mode, create a new instance
            return cls.initialize()
            
        with cls._lock:
            if cls._instance is None:
                return cls.initialize()
            return cls._instance

    @property
    def app_context(self) -> EmberAppContext:
        """
        Get the EmberAppContext instance, initializing if needed.

        Returns:
            The EmberAppContext instance

        Raises:
            RuntimeError: If accessed before initialization
        """
        if self._app_context is None:
            self._app_context = create_ember_app()
        return self._app_context

    @property
    def registry(self) -> ModelRegistry:
        """Access the model registry service."""
        return self.app_context.model_registry

    @property
    def config_manager(self) -> ConfigManager:
        """Access the configuration manager."""
        return self.app_context.config_manager

    @property
    def usage_service(self) -> Optional[UsageService]:
        """Access the usage tracking service."""
        return self.app_context.usage_service

    @property
    def logger(self) -> logging.Logger:
        """Access the logger instance."""
        return self.app_context.logger
        
    @classmethod
    def reset(cls) -> None:
        """
        Reset the EmberContext to its initial state.
        
        This is primarily useful for testing to ensure a clean state.
        """
        cls._instance = None
        if hasattr(cls._thread_local, 'instances'):
            delattr(cls._thread_local, 'instances')


def get_ember_context() -> EmberContext:
    """
    Get the global EmberContext instance, creating it if necessary.

    This is the recommended entry point for accessing the context.

    Returns:
        The global EmberContext instance
    """
    return EmberContext.get()


__all__ = [
    "EmberAppContext",
    "create_ember_app",
    "EmberContext",
    "get_ember_context",
]
