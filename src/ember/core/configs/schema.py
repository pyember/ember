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
    """Configuration for the model registry."""
    
    auto_discover: bool = True
    auto_register: bool = True
    cache_ttl: int = 3600
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    included_configs: List[str] = Field(default_factory=list)
    
    class Config:
        extra = "allow"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"


class DataPathsConfig(BaseModel):
    """Data paths configuration."""
    
    datasets: str = "datasets"
    cache: str = ".cache"
    
    class Config:
        extra = "allow"


class EmberConfig(BaseModel):
    """Main configuration schema for Ember."""
    
    model_registry: ModelRegistryConfig = Field(default_factory=ModelRegistryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data_paths: DataPathsConfig = Field(default_factory=DataPathsConfig)
    
    class Config:
        extra = "allow"


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