"""
Minimal test doubles for model registry components.

This module provides simplified test doubles that implement just enough functionality
to test client code without duplicating the implementation. Following the
principle of "avoid overmocking" from CLAUDE.md guidelines.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class MinimalProviderInfo:
    """Minimal test double for provider information."""

    name: str
    description: str = ""
    models: List[str] = field(default_factory=list)
    api_key: Optional[str] = None


@dataclass
class MinimalCost:
    """Minimal test double for cost information."""

    input_cost_per_thousand: float = 0.0
    output_cost_per_thousand: float = 0.0


@dataclass
class MinimalModelInfo:
    """Minimal test double for model information."""

    id: str
    name: str
    provider: str
    provider_model_id: Optional[str] = None
    cost: MinimalCost = field(default_factory=MinimalCost)

    @property
    def model_id(self) -> str:
        """Alias for id to maintain backward compatibility."""
        return self.id

    @property
    def model_name(self) -> str:
        """Alias for name to maintain backward compatibility."""
        return self.name

    def get_api_key(self) -> Optional[str]:
        """Get the API key for this model."""
        # In a real implementation, this would check app context
        return None


class MinimalModelRegistry:
    """Minimal test double for model registry."""

    def __init__(self):
        """Initialize with empty registry."""
        self.models: Dict[str, MinimalModelInfo] = {}
        self.providers: Dict[str, MinimalProviderInfo] = {}

    def register_model(self, model_info: MinimalModelInfo) -> None:
        """Register a model with the registry."""
        full_model_id = f"{model_info.provider}:{model_info.model_id}"
        self.models[full_model_id] = model_info

    def register_provider(self, provider_info: MinimalProviderInfo) -> None:
        """Register a provider with the registry."""
        self.providers[provider_info.name.lower()] = provider_info

    def list_models(self) -> List[str]:
        """List all registered model IDs."""
        return list(self.models.keys())

    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        return list(self.providers.keys())

    def get_model_info(self, model_id: str) -> MinimalModelInfo:
        """Get model information by ID."""
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found in registry")
        return self.models[model_id]

    def get_provider_info(self, provider_name: str) -> MinimalProviderInfo:
        """Get provider information by name."""
        lower_name = provider_name.lower()
        if lower_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not found in registry")
        return self.providers[lower_name]

    def discover_models(self) -> List[MinimalModelInfo]:
        """Simulated discovery of models."""
        # In a test double, just return already registered models
        return list(self.models.values())


class MinimalAnthropicModel:
    """Minimal test double for Anthropic model."""

    def __init__(self, model_id: str = "claude-3", api_key: Optional[str] = None):
        """Initialize with model ID and API key."""
        self.model_id = model_id
        self.api_key = api_key

    def generate(self, prompt: str) -> str:
        """Generate a response to a prompt."""
        # For testing, return a simple deterministic response
        return f"Response from {self.model_id}: {prompt[:20]}..."


class MinimalOpenAIModel:
    """Minimal test double for OpenAI model."""

    def __init__(self, model_id: str = "gpt-4", api_key: Optional[str] = None):
        """Initialize with model ID and API key."""
        self.model_id = model_id
        self.api_key = api_key

    def generate(self, prompt: str) -> str:
        """Generate a response to a prompt."""
        # For testing, return a simple deterministic response
        return f"Response from {self.model_id}: {prompt[:20]}..."


class MinimalModelFactory:
    """Minimal test double for model factory."""

    def __init__(self, registry: Optional[MinimalModelRegistry] = None):
        """Initialize with optional registry."""
        self.registry = registry or MinimalModelRegistry()

    def create_model(
        self, model_id: str
    ) -> Union[MinimalAnthropicModel, MinimalOpenAIModel]:
        """Create a model instance by ID."""
        # For testing, create a minimal model based on ID prefix
        if model_id.startswith("anthropic:"):
            return MinimalAnthropicModel(model_id.split(":")[1])
        elif model_id.startswith("openai:"):
            return MinimalOpenAIModel(model_id.split(":")[1])
        else:
            raise ValueError(f"Unknown model type: {model_id}")


# Export minimal test doubles
__all__ = [
    "MinimalProviderInfo",
    "MinimalCost",
    "MinimalModelInfo",
    "MinimalModelRegistry",
    "MinimalAnthropicModel",
    "MinimalOpenAIModel",
    "MinimalModelFactory",
]
