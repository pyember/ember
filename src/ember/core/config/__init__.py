"""Ember configuration system.

This package provides a minimal but extensible configuration system for Ember.
It allows loading configuration from YAML files and environment variables,
with support for validation, environment variable substitution, and clean
error handling.

Example usage:
```python
from ember.core.config import load_config

# Load configuration from default locations
config = load_config()

# Access configuration values
auto_discover = config.registry.auto_discover
logging_level = config.logging.level

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
```
"""

from .schema import EmberConfig, Provider, Model, Cost
from .loader import load_config
from .exceptions import ConfigError

__all__ = [
    'EmberConfig',
    'Provider',
    'Model',
    'Cost',
    'load_config',
    'ConfigError'
]