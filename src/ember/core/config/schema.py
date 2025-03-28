"""Configuration schema module.

This module defines the data structures used for configuration in Ember.
The schemas are designed to be minimal but extensible through Pydantic.
"""

from typing import Dict, Optional, Union

from pydantic import BaseModel, Field, computed_field


class Cost(BaseModel):
    """Value object for model cost calculations."""

    input_cost_per_thousand: float = 0.0
    output_cost_per_thousand: float = 0.0

    def calculate(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Calculated cost in currency units
        """
        return (
            self.input_cost_per_thousand * input_tokens
            + self.output_cost_per_thousand * output_tokens
        ) / 1000


class RateLimit(BaseModel):
    """Rate limit configuration for models."""

    tokens_per_minute: int = 0
    requests_per_minute: int = 0

    model_config = {"extra": "allow"}


class Model(BaseModel):
    """Model configuration with minimal required fields.

    This class captures the essential configuration for a language model,
    while allowing arbitrary additional fields through Pydantic's extra="allow".

    Attributes:
        id: Unique identifier for the model
        name: Human-readable name of the model
        provider: Provider name (optional - inferred from parent provider if missing)
        cost_input: Cost per 1000 input tokens
        cost_output: Cost per 1000 output tokens
        rate_limit: Optional rate limiting configuration
    """

    # Required fields with flexible validation for backward compatibility
    id: str
    name: str
    provider: Optional[str] = None  # Can be inferred from parent provider

    # Optional fields with defaults
    cost_input: float = 0.0
    cost_output: float = 0.0
    rate_limit: Optional[Union[Dict[str, int], RateLimit]] = None

    # Allow arbitrary extension
    model_config = {"extra": "allow"}

    @computed_field
    def cost(self) -> Cost:
        """Get cost calculator for this model."""
        # We can pass the input and output costs as named parameters
        return Cost(
            input_cost_per_thousand=self.cost_input,
            output_cost_per_thousand=self.cost_output,
        )

    def __init__(self, **data):
        # Convert rate_limit dictionary to RateLimit model if it's a dict
        if "rate_limit" in data and isinstance(data["rate_limit"], dict):
            data["rate_limit"] = RateLimit(**data["rate_limit"])
        super().__init__(**data)


class Provider(BaseModel):
    """Provider configuration with minimal required fields.

    This class captures the essential configuration for a model provider
    like OpenAI or Anthropic, while allowing arbitrary extension.

    Attributes:
        enabled: Whether this provider is enabled
        api_key: API key for authentication (falls back to environment)
        api_keys: Dictionary of API keys for different environments/purposes
        models: Dictionary of model configurations keyed by name or list of models
    """

    # Fields
    enabled: bool = True
    api_key: Optional[str] = None
    api_keys: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    models: Dict[str, Model] = Field(default_factory=dict)

    # Allow arbitrary extension
    model_config = {"extra": "allow"}

    def get_model_config(self, model_id: str) -> Optional[Model]:
        """Get configuration for a specific model.

        Args:
            model_id: Model identifier (with or without provider prefix)

        Returns:
            Model configuration if found, None otherwise
        """
        # Check if model is explicitly configured
        for key, model in self.models.items():
            if key == model_id or f"{self.__root_key__}:{key}" == model_id:
                return model
        return None

    def __init__(self, **data):
        # Convert models list to dictionary if it's provided as a list
        if "models" in data and isinstance(data["models"], list):
            models_list = data["models"]
            models_dict = {model.id: model for model in models_list}
            data["models"] = models_dict

        super().__init__(**data)
        # Store the root key this provider was initialized with
        self.__root_key__ = ""


class RegistryConfig(BaseModel):
    """Configuration for the model registry.

    Attributes:
        auto_discover: Whether to automatically discover models
        providers: Dictionary of provider configurations keyed by name
    """

    auto_discover: bool = True
    providers: Dict[str, Provider] = Field(default_factory=dict)

    # Allow arbitrary extension
    model_config = {"extra": "allow"}

    def __init__(self, **data):
        super().__init__(**data)
        # Set the root key for each provider to its dictionary key
        for key, provider in self.providers.items():
            provider.__root_key__ = key


class LoggingConfig(BaseModel):
    """Configuration for logging.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """

    level: str = "INFO"

    # Allow arbitrary extension
    model_config = {"extra": "allow"}


class EmberConfig(BaseModel):
    """Root configuration with minimal required sections.

    This is the top-level configuration class for Ember.

    Attributes:
        registry: Model registry configuration
        logging: Logging configuration
    """

    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Allow arbitrary extension
    model_config = {"extra": "allow"}

    def get_provider(self, name: str) -> Optional[Provider]:
        """Get provider by name.

        Args:
            name: Provider name (case-insensitive)

        Returns:
            Provider configuration if found, None otherwise
        """
        return self.registry.providers.get(name.lower())

    def get_model_config(self, model_id: str) -> Optional[Model]:
        """Get model configuration by ID.

        Args:
            model_id: Model identifier in "provider:model" format

        Returns:
            Model configuration if found, None otherwise
        """
        if ":" in model_id:
            provider_name, model_name = model_id.split(":", 1)
            provider = self.get_provider(provider_name)
            if provider:
                return provider.get_model_config(model_name)
        return None


class EmberSettings(EmberConfig):
    """Configuration settings for Ember.

    This class extends EmberConfig to provide a more user-friendly API for configuration.
    It resolves circular import issues with model configuration.

    Usage:
        settings = EmberSettings()
        settings.registry.auto_discover = True
    """

    pass
