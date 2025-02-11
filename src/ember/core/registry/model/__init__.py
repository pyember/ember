"""Ember Core Model Registry Initialization.

This module aggregates and exposes the core components for interacting with Ember's
model registry. It offers a streamlined API for initializing the registry, retrieving model
instances, and leveraging advanced services such as usage tracking. For convenience, the
following symbols can be imported:

    from ember import initialize_ember, load_model

Initialization (via initialize_ember()):
    - Merges the primary YAML configuration with auxiliary configuration files.
    - Resolves environment variable placeholders.
    - Instantiates a new ModelRegistry.
    - Optionally performs model discovery.

After initialization, models can be retrieved using load_model() or via the advanced
ModelService, which supports usage tracking and custom provider parameters.

Exposed Public Symbols:
    Model Schemas and Services:
        * ModelInfo, ProviderInfo, ModelCost, RateLimit, ChatRequest, ChatResponse, etc.
        * ModelService (with advanced capabilities)
        * UsageService

    Initialization & Configuration:
        * EmberSettings, initialize_ember
        * load_model, get_service (aliases for simplified usage)

    Provider Utilities:
        * ModelFactory, discover_providers, ModelEnum, parse_model_str
"""

from __future__ import annotations

from typing import List, Optional

# Core schemas
from ember.core.registry.model.schemas.model_info import ModelInfo
from ember.core.registry.model.schemas.provider_info import ProviderInfo
from ember.core.registry.model.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.schemas.chat_schemas import ChatRequest, ChatResponse
from ember.core.registry.model.providers.base_provider import (
    BaseChatParameters,
    BaseProviderModel,
)
from ember.core.registry.model.schemas.usage import (
    UsageStats,
    UsageRecord,
    UsageSummary,
)

# Core services
from ember.core.registry.model.services.model_service import ModelService
from ember.core.registry.model.services.usage_service import UsageService

# Configuration and initialization
from ember.core.registry.model.config.settings import EmberSettings, initialize_ember

# Registry and factory utilities
from ember.core.registry.model.registry.model_registry import ModelRegistry
from ember.core.registry.model.registry import ModelFactory, discover_providers
from ember.core.registry.model.registry.model_enum import ModelEnum, parse_model_str


def load_model(*, model_id: str, registry: ModelRegistry) -> BaseProviderModel:
    """Retrieve a provider model instance from the given registry.

    Args:
        model_id (str): Unique identifier for the model (e.g., "openai:gpt-4o").
        registry (ModelRegistry): Instance of ModelRegistry, typically obtained via
            initialize_ember().

    Returns:
        BaseProviderModel: The resolved provider model instance ready for invocation.

    Raises:
        ValueError: If the provided registry is None.
    """
    if registry is None:
        raise ValueError("Registry is not initialized. Please call 'initialize_ember()' first.")
    return registry.get_model(model_id=model_id)


def get_service(*, usage_tracking: bool = False, registry: Optional[ModelRegistry] = None) -> ModelService:
    """Obtain a ModelService instance configured for advanced operations.

    If no registry is provided, a new ModelRegistry is automatically initialized using
    initialize_ember().

    Args:
        usage_tracking (bool): Flag indicating whether to enable usage tracking. Defaults to False.
        registry (Optional[ModelRegistry]): An instance of ModelRegistry. If None, it will be auto-initialized.

    Returns:
        ModelService: A ModelService instance initialized with the registry and optional usage tracking.
    """
    if registry is None:
        registry = initialize_ember()
    usage_service: Optional[UsageService] = UsageService() if usage_tracking else None
    return ModelService(registry=registry, usage_service=usage_service)


def initialize_ember_service(*, usage_tracking: bool = False, config_path: Optional[str] = None) -> ModelService:
    """Initialize Ember and return a fully configured ModelService instance.

    This convenience function initializes the Ember registry using the merged configuration and
    creates a ModelService with optional usage tracking.

    Args:
        usage_tracking (bool): If True, a UsageService is attached for usage logging. Defaults to False.
        config_path (Optional[str]): Custom path to the YAML configuration file. If None, the default configuration is used.

    Returns:
        ModelService: A fully configured service instance ready for model invocation.
    """
    registry: ModelRegistry = initialize_ember(config_path=config_path)
    usage_service: Optional[UsageService] = UsageService() if usage_tracking else None
    return ModelService(registry=registry, usage_service=usage_service)


# Public API alias.
get_model = load_model

__all__: List[str] = [
    "ModelInfo",
    "ProviderInfo",
    "ModelCost",
    "RateLimit",
    "ChatRequest",
    "ChatResponse",
    "BaseChatParameters",
    "UsageStats",
    "UsageRecord",
    "UsageSummary",
    "ModelService",
    "UsageService",
    "EmberSettings",
    "initialize_ember",
    "ModelFactory",
    "discover_providers",
    "ModelEnum",
    "parse_model_str",
    "load_model",
    "get_service",
    "get_model",
    "initialize_ember_service",
]
