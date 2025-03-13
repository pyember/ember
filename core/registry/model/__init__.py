"""Ember Core Model Registry Initialization.

This module provides the core components for model registry functionality.
"""

from __future__ import annotations
from typing import List

# Absolute imports for core schemas
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)
from ember.core.registry.model.base.schemas.usage import (
    UsageStats,
    UsageRecord,
    UsageSummary,
)

# Base provider classes
from ember.core.registry.model.providers.base_provider import (
    BaseChatParameters,
    BaseProviderModel,
)

# Registry components
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.registry.factory import ModelFactory
from ember.core.registry.model.config.model_enum import (
    ModelEnum,
    parse_model_str,
)

# Services
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.base.services.usage_service import UsageService

# Configuration and initialization - moved to avoid circular imports
from ember.core.registry.model.config.settings import EmberSettings

# Import key components
from .base.services.model_service import ModelService
from .base.utils.model_registry_exceptions import (
    ModelRegistrationError,
    ModelDiscoveryError,
)

# Absolute imports for exceptions
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    ModelRegistrationError,
    ModelDiscoveryError,
)

# Add ModelRegistry import
from ember.core.registry.model.base.registry.model_registry import ModelRegistry


# Add load_model function
def load_model(model_id: str, registry: ModelRegistry) -> BaseProviderModel:
    """Public helper to load model instances from registry.

    Args:
        model_id (str): Model identifier string
        registry (ModelRegistry): ModelRegistry instance to query

    Returns:
        Instantiated provider model
    """
    return registry.get_model(model_id)


__all__: List[str] = [
    # Schemas
    "ModelInfo",
    "ProviderInfo",
    "ModelCost",
    "RateLimit",
    "ChatRequest",
    "ChatResponse",
    "UsageStats",
    "UsageRecord",
    "UsageSummary",
    # Base classes
    "BaseChatParameters",
    "BaseProviderModel",
    # Registry
    "ModelRegistry",
    "ModelFactory",
    "ModelEnum",
    "parse_model_str",
    # Services
    "ModelService",
    "UsageService",
    # Settings
    "EmberSettings",
    "initialize_ember",
    "ModelRegistrationError",
    "ModelDiscoveryError",
    "load_model",
]


# Initialization function - defined here to avoid circular imports
def initialize_ember(
    config_path: str | None = None,
    auto_register: bool = True,
    auto_discover: bool = True,
) -> ModelRegistry:
    """Initialize the Ember model registry.

    Args:
        config_path (str | None): Optional path to config file
        auto_register (bool): Automatically register models from config
        auto_discover (bool): Enable provider model discovery

    Returns:
        Initialized ModelRegistry instance
    """
    from ember.core.registry.model.config.settings import initialize_ember as _init

    return _init(
        config_path=config_path,
        auto_register=auto_register,
        auto_discover=auto_discover,
    )
