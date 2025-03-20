"""Ember configuration system.

This package provides a standardized configuration system for Ember.
It allows loading configuration from YAML files and environment variables,
with support for validation, environment variable substitution, and thread-safe
configuration management.

Example usage:
```python
from ember.core.config import load_config, create_config_manager

# Load configuration directly
config = load_config(config_path="config.yaml")

# Access configuration values
auto_discover = config.registry.auto_discover
logging_level = config.logging.level

# Or use the config manager for more features
config_manager = create_config_manager(config_path="config.yaml")
config = config_manager.get_config()

# Get provider by name
provider = config.get_provider("openai")
if provider and provider.enabled:
    # Use provider
    pass

# Get model configuration
model = config.get_model_config("openai:gpt-4")
if model:
    # Use model configuration
    cost = model.cost.calculate(100, 200)  # Calculate cost for tokens

# Set API key
config_manager.set_provider_api_key("openai", "sk-your-key")
```
"""

from .exceptions import ConfigError
from .loader import load_config, merge_dicts, resolve_env_vars
from .manager import ConfigManager, create_config_manager
from .schema import Cost, EmberConfig, LoggingConfig, Model, Provider, RegistryConfig

__all__ = [
    # Schema classes
    "EmberConfig",
    "Provider",
    "Model",
    "Cost",
    "RegistryConfig",
    "LoggingConfig",
    # Loader functions
    "load_config",
    "merge_dicts",
    "resolve_env_vars",
    # Manager classes
    "ConfigManager",
    "create_config_manager",
    # Exceptions
    "ConfigError",
]
