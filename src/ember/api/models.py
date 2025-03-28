"""Models API for Ember.

This module provides a clean, type-safe API for working with language models from
various providers. It offers a consistent interface with multiple invocation patterns
to suit different programming styles.

Examples:
    # 1. Initialize the model system
    from ember.api import models

    # 2. Use namespace-style access (preferred pattern)
    response = models.openai.gpt4o("What is the capital of France?")

    # 3. Use with configuration builder
    from ember.api.models import ModelBuilder

    model = (
        ModelBuilder()
        .temperature(0.7)
        .max_tokens(100)
        .build("anthropic:claude-3-5-sonnet")
    )
    response = model.generate(prompt="Explain quantum computing")

    # 4. Use type-safe ModelEnum references
    from ember.api.models import ModelEnum, ModelAPI

    model = ModelAPI.from_enum(ModelEnum.OPENAI_GPT4O)
    response = model.generate(prompt="What's the best programming language?")

    # 5. Access the underlying registry directly for advanced usage
    from ember.api.models import get_registry

    registry = get_registry()
    available_models = registry.list_models()

    # 6. Track model usage with UsageService
    from ember.api.models import get_usage_service

    usage = get_usage_service()
    stats = usage.get_usage_stats()
    print(f"Total tokens: {stats.total_tokens}")
"""

from typing import Any, Dict, Optional, Union

from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit

# Model schemas and types
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.schemas.usage import UsageStats

# Service and provider model access
from ember.core.registry.model.base.services.model_service import (
    ModelService,
)
from ember.core.registry.model.base.services.usage_service import UsageService

# Configuration and enums
from ember.core.registry.model.config.model_enum import ModelEnum

# Core initialization and registry access
from ember.core.registry.model.initialization import initialize_registry

# Language model module classes
from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig
from ember.core.registry.model.providers.base_provider import BaseProviderModel

# Singleton instances for global access
_registry: Optional[ModelRegistry] = None
_model_service: Optional[ModelService] = None
_usage_service: Optional[UsageService] = None


class ModelAPI:
    """High-level API for interacting with a specific model."""

    def __init__(self, model_id: str) -> None:
        """Initialize the model API.

        Args:
            model_id: The model identifier
        """
        self.model_id = model_id
        self._model = get_model_service().get_model(model_id)

    def generate(self, prompt: str, **kwargs: Any) -> ChatResponse:
        """Generate a response from the model.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters to pass to the model

        Returns:
            The model response
        """
        return self._model(prompt=prompt, **kwargs)

    @classmethod
    def from_enum(cls, model_enum: Any) -> "ModelAPI":
        """Create a ModelAPI instance from a ModelEnum value.

        Args:
            model_enum: The model enum value

        Returns:
            ModelAPI instance
        """
        # Extract the string value from the enum
        if hasattr(model_enum, "value"):
            return cls(model_id=model_enum.value)
        return cls(model_id=str(model_enum))


class ModelBuilder:
    """Builder pattern for configuring model parameters."""

    def __init__(self) -> None:
        """Initialize the model builder."""
        self._config: Dict[str, Any] = {}

    def temperature(self, value: float) -> "ModelBuilder":
        """Set the temperature parameter.

        Args:
            value: Temperature value (0.0-1.0)

        Returns:
            Self for chaining
        """
        self._config["temperature"] = value
        return self

    def max_tokens(self, value: int) -> "ModelBuilder":
        """Set the maximum tokens parameter.

        Args:
            value: Maximum number of tokens

        Returns:
            Self for chaining
        """
        self._config["max_tokens"] = value
        return self

    def timeout(self, value: int) -> "ModelBuilder":
        """Set the timeout parameter.

        Args:
            value: Timeout in seconds

        Returns:
            Self for chaining
        """
        self._config["timeout"] = value
        return self

    def build(self, model_id: Union[str, Any]) -> ModelAPI:
        """Build a model API instance with the configured parameters.

        Args:
            model_id: The model identifier or enum

        Returns:
            Configured ModelAPI instance
        """
        # Extract string value if it's an enum
        model_id_str = model_id
        if hasattr(model_id, "value"):
            model_id_str = model_id.value

        api = ModelAPI(model_id=str(model_id_str))

        # Add the configuration to the model
        for key, value in self._config.items():
            setattr(api._model, key, value)

        return api


def get_registry() -> ModelRegistry:
    """Get the global model registry, initializing it if necessary.

    Returns:
        The global model registry
    """
    global _registry
    if _registry is None:
        _registry = initialize_registry(auto_discover=True)
    return _registry


def get_model_service() -> ModelService:
    """Get the global model service, initializing it if necessary.

    Returns:
        The global model service
    """
    global _model_service, _registry
    if _model_service is None:
        registry = get_registry()
        _model_service = ModelService(registry=registry)
    return _model_service


def get_usage_service() -> UsageService:
    """Get the global usage service, initializing it if necessary.

    Returns:
        The global usage service
    """
    global _usage_service, _registry
    if _usage_service is None:
        # Create the usage service
        _usage_service = UsageService()
    return _usage_service


class _ProviderNamespace:
    """Provider namespace for accessing models by name."""

    def __init__(self, provider_name: str) -> None:
        """Initialize the provider namespace.

        Args:
            provider_name: The name of the provider
        """
        self._provider_name = provider_name
        self._registry = get_registry()
        self._service = get_model_service()

    def __getattr__(self, model_name: str) -> Any:
        """Get a callable for invoking a model.

        Args:
            model_name: The name of the model

        Returns:
            A callable that invokes the model

        Raises:
            AttributeError: If the model doesn't exist
        """
        model_id = f"{self._provider_name}:{model_name}"

        # Check if model exists
        if not self._registry.is_registered(model_id):
            model_id_normalized = model_id.replace("_", "-")
            if self._registry.is_registered(model_id_normalized):
                model_id = model_id_normalized
            else:
                # Create a stub function instead of raising error
                def stub_invoke(prompt: str, **kwargs: Any) -> ChatResponse:
                    raise ValueError(
                        f"Model {model_id} not found or API key not configured. "
                        f"Make sure the model exists and you have set the appropriate API key."
                    )

                stub_invoke.__doc__ = f"""Invoke the {model_id} model.
                
                Note: This model is not currently available. Make sure you have the appropriate API key set.
                
                Args:
                    prompt: The input prompt
                    **kwargs: Additional parameters to pass to the model
                    
                Raises:
                    ValueError: When the model is not available
                """
                return stub_invoke

        # Get the model
        model = self._service.get_model(model_id)

        # Return a callable that invokes the model
        def invoke(prompt: str, **kwargs: Any) -> ChatResponse:
            return model(prompt=prompt, **kwargs)

        # Add model metadata
        try:
            model_info = self._registry.get_model_info(model_id)
            model_name = model_info.model_name if model_info else model_id
            invoke.__doc__ = f"""Invoke the {model_name} model.
            
            Args:
                prompt: The input prompt
                **kwargs: Additional parameters to pass to the model
                
            Returns:
                The model response
            """
        except Exception:
            # Fallback if we can't get model info
            invoke.__doc__ = f"""Invoke the {model_id} model.
            
            Args:
                prompt: The input prompt
                **kwargs: Additional parameters to pass to the model
                
            Returns:
                The model response
            """

        return invoke


# Create provider namespaces
openai = _ProviderNamespace("openai")
anthropic = _ProviderNamespace("anthropic")
deepmind = _ProviderNamespace("deepmind")


# Aliases for common models - define these conditionally
# instead of raising errors during import
def _get_model_or_stub(namespace: Any, model_name: str) -> Any:
    """Get a model from namespace or return a stub function if not available.

    Args:
        namespace: Provider namespace
        model_name: Model name

    Returns:
        Model function or stub that raises a clear error
    """
    try:
        return getattr(namespace, model_name)
    except (AttributeError, ValueError):
        # Return a stub function instead
        def stub(*args: Any, **kwargs: Any) -> Any:
            """Stub function that raises an error when called."""
            raise ValueError(
                f"Model {namespace._provider_name}:{model_name} not found or API key not configured."
            )

        return stub


# Create the aliases
gpt4 = _get_model_or_stub(openai, "gpt4")
gpt4o = _get_model_or_stub(openai, "gpt4o")
claude = _get_model_or_stub(anthropic, "claude_3_5_sonnet")
gemini = _get_model_or_stub(deepmind, "gemini_1_5_pro")

# Export public API
__all__ = [
    # Primary initialization and access
    "get_registry",
    "get_model_service",
    "get_usage_service",
    "ModelAPI",
    "ModelBuilder",
    # Provider namespaces
    "openai",
    "anthropic",
    "deepmind",
    # Model aliases
    "gpt4",
    "gpt4o",
    "claude",
    "gemini",
    # Core classes and types for advanced usage
    "ModelRegistry",
    "ModelService",
    "UsageService",
    "BaseProviderModel",
    "LMModule",
    "LMModuleConfig",
    # Model information and configuration
    "ModelInfo",
    "ProviderInfo",
    "ModelCost",
    "RateLimit",
    "UsageStats",
    # Request and response types
    "ChatRequest",
    "ChatResponse",
    # Constants and enums
    "ModelEnum",
]
