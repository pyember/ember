"""Models API for Ember.

This module provides access to language models from various providers
with a simple, consistent interface. It focuses on practical, proven patterns
for working with LLMs in production settings.

Examples:
    # 1. Initialize the system (only needed once)
    from ember.api.models import initialize_registry
    
    registry = initialize_registry(auto_discover=True)
    
    # 2. Create a model service (the main entry point)
    from ember.api.models import create_model_service
    
    model_service = create_model_service(registry=registry)
    
    # 3. Call a model directly
    response = model_service.invoke_model(
        model_id="openai:gpt-4",
        prompt="What is machine learning?",
        temperature=0.7
    )
    print(response.text)
    
    # 4. Use the registry to discover and inspect models
    available_models = registry.list_models()
    
    for model_id in available_models:
        model_info = registry.get_model_info(model_id)
        print(f"{model_id}: {model_info.name} - Context: {model_info.context_window}")
        
    # 5. Get a model directly and use it
    model = registry.get_model("anthropic:claude-3-sonnet")
    response = model(prompt="Explain quantum computing", temperature=0.5)
    
    # 6. Register a custom model
    from ember.api.models import ModelInfo, ModelCost
    
    custom_model = ModelInfo(
        id="custom:my-model",
        name="My Custom Model",
        cost=ModelCost(
            input_cost_per_thousand=0.001,
            output_cost_per_thousand=0.002
        ),
        context_window=32000,
        provider={
            "name": "CustomProvider",
            "api_base": "https://api.example.com/v1"
        }
    )
    
    registry.register_model(model_info=custom_model)
"""

# Core initialization and registry
from ember.core.registry.model.initialization import initialize_registry
from ember.core.registry.model.base.registry.model_registry import ModelRegistry

# Service and provider model access
from ember.core.registry.model.base.services.model_service import (
    ModelService,
    create_model_service,
)
from ember.core.registry.model.base.services.usage_service import UsageService
from ember.core.registry.model.providers.base_provider import BaseProviderModel

# Model schemas and types
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.base.schemas.usage import UsageStats
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatResponse as ModelResponse,
)

# Configuration and enums
from ember.core.registry.model.config.model_enum import ModelEnum

__all__ = [
    # Primary initialization and access
    "initialize_registry",  # Initialize the model registry
    "create_model_service",  # Create model service for easier access
    # Core classes
    "ModelRegistry",  # Registry for model management
    "ModelService",  # Service for invoking models
    "UsageService",  # Service for tracking usage
    "BaseProviderModel",  # Base class for provider models
    # Model information and configuration
    "ModelInfo",  # Model metadata
    "ModelCost",  # Cost information
    "RateLimit",  # Rate limiting configuration
    "UsageStats",  # Usage statistics
    # Response types
    "ModelResponse",  # Standard response format
    # Constants and enums
    "ModelEnum",  # Type-safe model references
]
