"""Configuration schema module.

This module defines Pydantic models for validating Ember configuration data.
These models provide type checking, validation, and documentation of the
configuration structure.
"""

from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field


class ApiKeyConfig(BaseModel):
    """API key configuration for a provider."""
    
    key: str = ""
    org_id: Optional[str] = None
    endpoint: Optional[str] = None
    
    class Config:
        extra = "allow"


class CostConfig(BaseModel):
    """Cost configuration for a model.
    
    This schema handles both per thousand and per million pricing formats
    for compatibility with different provider documentation styles.
    Only one format needs to be specified; the other will be derived.
    
    Attributes:
        input_cost_per_thousand: Cost per 1000 input tokens
        output_cost_per_thousand: Cost per 1000 output tokens
        input_cost_per_million: Cost per 1 million input tokens
        output_cost_per_million: Cost per 1 million output tokens
    """
    
    input_cost_per_thousand: float = 0.0
    output_cost_per_thousand: float = 0.0
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    
    def model_post_init(self, __context):
        """Derive missing values if only one cost format is provided."""
        # If per_million is provided but not per_thousand
        if self.input_cost_per_million and not self.input_cost_per_thousand:
            self.input_cost_per_thousand = self.input_cost_per_million / 1000
        
        if self.output_cost_per_million and not self.output_cost_per_thousand:
            self.output_cost_per_thousand = self.output_cost_per_million / 1000
            
        # If per_thousand is provided but not per_million
        if self.input_cost_per_thousand and not self.input_cost_per_million:
            self.input_cost_per_million = self.input_cost_per_thousand * 1000
            
        if self.output_cost_per_thousand and not self.output_cost_per_million:
            self.output_cost_per_million = self.output_cost_per_thousand * 1000
    
    class Config:
        extra = "allow"


class RateLimitConfig(BaseModel):
    """Rate limit configuration for a model.
    
    Attributes:
        tokens_per_minute: Maximum tokens allowed per minute
        requests_per_minute: Maximum requests allowed per minute
        max_batch_size: Maximum number of prompts in a batch request (optional)
        max_context_length: Maximum context length in tokens (optional)
    """
    
    tokens_per_minute: int = 0
    requests_per_minute: int = 0
    max_batch_size: Optional[int] = None
    max_context_length: Optional[int] = None
    
    class Config:
        extra = "allow"


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    
    id: str
    name: str
    provider: str
    cost: CostConfig = Field(default_factory=CostConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"


class ProviderConfig(BaseModel):
    """Configuration for a model provider."""
    
    enabled: bool = True
    api_keys: Dict[str, ApiKeyConfig] = Field(default_factory=dict)
    timeout: float = 30.0
    max_retries: int = 3
    models: List[ModelConfig] = Field(default_factory=list)
    
    class Config:
        extra = "allow"


class ModelRegistryConfig(BaseModel):
    """Configuration for the model registry.
    
    This centralized schema handles all model registry configuration including
    discovery settings and provider configurations.
    
    Attributes:
        auto_discover: Whether to automatically discover models from provider APIs
        auto_register: Whether to automatically register models from configuration
        cache_ttl: Time-to-live for discovery cache in seconds
        providers: Dictionary of provider configurations keyed by name
        included_configs: List of additional config files to include
    """
    
    auto_discover: bool = True
    auto_register: bool = True
    cache_ttl: int = 3600
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    included_configs: List[str] = Field(default_factory=list)
    
    model_config = {"extra": "allow"}
    
    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Get provider by name.
        
        Args:
            name: Provider name (case-insensitive)
            
        Returns:
            Provider configuration if found, None otherwise
        """
        return self.providers.get(name.lower())
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
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
                for model in provider.models:
                    if model.id == model_name or model.id == model_id:
                        return model
        return None


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    model_config = {"extra": "allow"}


class DataPathsConfig(BaseModel):
    """Data paths configuration."""
    
    datasets: str = "datasets"
    cache: str = ".cache"
    
    model_config = {"extra": "allow"}


class EmberConfig(BaseModel):
    """Main configuration schema for Ember.
    
    This is the root configuration class that contains all settings for the
    Ember framework, including model registry, logging, and data paths.
    
    Attributes:
        model_registry: Model registry configuration
        logging: Logging configuration
        data_paths: Data paths configuration
    """
    
    model_registry: ModelRegistryConfig = Field(default_factory=ModelRegistryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data_paths: DataPathsConfig = Field(default_factory=DataPathsConfig)
    
    model_config = {"extra": "allow"}
    
    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Get provider by name.
        
        Args:
            name: Provider name (case-insensitive)
            
        Returns:
            Provider configuration if found, None otherwise
        """
        return self.model_registry.get_provider(name)
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration by ID.
        
        Args:
            model_id: Model identifier in "provider:model" format
            
        Returns:
            Model configuration if found, None otherwise
        """
        return self.model_registry.get_model_config(model_id)


# Default configuration used when no configuration file exists
DEFAULT_CONFIG: Dict[str, Any] = {
    "model_registry": {
        "auto_discover": True,
        "auto_register": True,
        "cache_ttl": 3600,
        "providers": {
            "openai": {
                "enabled": True,
                "api_keys": {"default": {"key": "${OPENAI_API_KEY}"}},
            },
            "anthropic": {
                "enabled": True,
                "api_keys": {"default": {"key": "${ANTHROPIC_API_KEY}"}},
            },
            "google": {
                "enabled": True,
                "api_keys": {"default": {"key": "${GOOGLE_API_KEY}"}},
            },
        },
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
    "data_paths": {
        "datasets": "datasets",
        "cache": ".cache",
    },
}