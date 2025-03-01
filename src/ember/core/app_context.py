import logging
from typing import Optional, Any, Dict

from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.base.services.usage_service import UsageService
from ember.core.registry.model.base.registry.model_registry import ModelRegistry


class EmberAppContext:
    """
    Application context for Ember, holding references to core services.

    This context serves as the composition root for dependency injection, eliminating
    reliance on global singleton state. For architectural details, see ARCHITECTURE.md.
    """

    def __init__(
        self,
        config_manager: "Any",  # kept generic; its concrete type comes from config.
        model_registry: ModelRegistry,
        usage_service: Optional[UsageService],
        logger: logging.Logger,
    ) -> None:
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
    """Global application context placeholder."""

    pass


__all__ = [
    "EmberAppContext",
    "create_ember_app",
    "EmberContext",
]
