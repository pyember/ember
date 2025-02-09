"""Ember Core Model Registry Initialization.

This module aggregates and exposes the core components utilized within the Ember core
model registry. These components include schemas, services, configuration settings,
as well as registry and factory utilities.

Exposed Components:
  - Core Schemas:
      • ModelInfo: Schema for model metadata.
      • ProviderInfo: Schema for provider information.
      • ModelCost, RateLimit: Schemas for model cost and rate limiting.
      • ChatRequest, ChatResponse, BaseChatParameters: Schemas for chat operations.
      • UsageStats, UsageRecord, UsageSummary: Schemas for usage monitoring.
  - Core Services:
      • ModelService: Service responsible for model operations.
      • UsageService: Service responsible for tracking usage.
  - Configuration & Initialization:
      • EmberSettings: Application configuration settings.
      • initialize_global_registry: Function for initializing the global registry.
  - Registry & Factory Utilities:
      • ModelRegistry, GLOBAL_MODEL_REGISTRY: The model registry class and its global instance.
      • ModelFactory, discover_providers: Utilities for model instantiation and provider discovery.
      • ModelEnum, parse_model_str: Tools for handling and validating model enumerations.
"""

from __future__ import annotations

# Core schemas:
from .core.schemas.model_info import ModelInfo
from .core.schemas.provider_info import ProviderInfo
from .core.schemas.cost import ModelCost, RateLimit
from .core.schemas.chat_schemas import ChatRequest, ChatResponse, BaseChatParameters
from .core.schemas.usage import UsageStats, UsageRecord, UsageSummary

# Core services:
from .core.services.model_service import ModelService
from .core.services.usage_service import UsageService

# Configuration and initialization:
from .settings import EmberSettings, initialize_global_registry

# Registry and factory utilities:
from .model_registry import ModelRegistry, GLOBAL_MODEL_REGISTRY
from .factory import ModelFactory, discover_providers
from .model_enum import ModelEnum, parse_model_str

from typing import List

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
    "initialize_global_registry",
    "ModelRegistry",
    "GLOBAL_MODEL_REGISTRY",
    "ModelFactory",
    "discover_providers",
    "ModelEnum",
    "parse_model_str",
]
