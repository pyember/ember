"""Ember configuration system.

This package provides a unified configuration system for Ember, supporting:
- Multiple configuration sources (files, environment variables)
- Schema validation with Pydantic
- Thread-safe access and updates
- Environment variable resolution
- Configuration transformations

The primary entry point is the ConfigManager class and the create_default_config_manager
function, which returns a pre-configured manager for common Ember use cases.

Example usage:
```python
from ember.core.configs import create_default_config_manager

# Create a default manager
config_manager = create_default_config_manager()

# Load configuration
config = config_manager.get_config()

# Access configuration values
auto_discover = config.model_registry.auto_discover
```
"""

from .exceptions import ConfigError
from .providers import ConfigProvider, YamlFileProvider, EnvironmentProvider
from .transformer import ConfigTransformer, deep_merge, resolve_env_vars
from .schema import (
    EmberConfig, ModelRegistryConfig, ProviderConfig, 
    ModelConfig, ApiKeyConfig, DEFAULT_CONFIG
)
from .config_manager import ConfigManager, create_default_config_manager

__all__ = [
    # Exceptions
    "ConfigError",
    
    # Providers
    "ConfigProvider",
    "YamlFileProvider",
    "EnvironmentProvider",
    
    # Transformers
    "ConfigTransformer",
    "deep_merge",
    "resolve_env_vars",
    
    # Schema
    "EmberConfig",
    "ModelRegistryConfig",
    "ProviderConfig", 
    "ModelConfig",
    "ApiKeyConfig",
    "DEFAULT_CONFIG",
    
    # Manager
    "ConfigManager",
    "create_default_config_manager",
]