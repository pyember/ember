"""Ember Core Model Registry Initialization.

This module provides the core components for model registry functionality.
"""

from __future__ import annotations
from typing import List

# Absolute imports for core schemas
from src.ember.core.registry.model.base.schemas.model_info import ModelInfo
from src.ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from src.ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from src.ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)
from src.ember.core.registry.model.base.schemas.usage import (
    UsageStats,
    UsageRecord,
    UsageSummary,
)

# Base provider classes
from src.ember.core.registry.model.providers.base_provider import (
    BaseChatParameters,
    BaseProviderModel,
)

# Registry components
from src.ember.core.registry.model.base.registry.model_registry import ModelRegistry
from src.ember.core.registry.model.base.registry.factory import ModelFactory
from src.ember.core.registry.model.base.registry.model_enum import (
    ModelEnum,
    parse_model_str,
)

# Services
from src.ember.core.registry.model.base.services.model_service import ModelService
from src.ember.core.registry.model.base.services.usage_service import UsageService

# Configuration and initialization - moved to avoid circular imports
from src.ember.core.registry.model.config.settings import EmberSettings

# Import key components
from .base.services.model_service import ModelService
from .base.utils.model_registry_exceptions import (
    ModelRegistrationError,
    ModelDiscoveryError,
)

# Absolute imports for exceptions
from src.ember.core.registry.model.base.utils.model_registry_exceptions import (
    ModelRegistrationError,
    ModelDiscoveryError,
)

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
]


# Initialization function - defined here to avoid circular imports
def initialize_ember(config_path: str | None = None) -> ModelRegistry:
    """Initialize the Ember model registry.

    Args:
        config_path: Optional path to config file. If None, uses default.

    Returns:
        Initialized ModelRegistry instance.
    """
    from src.ember.core.registry.model.base.registry.model_registry import ModelRegistry

    return ModelRegistry.initialize(config_path)
