# Ember Configuration System

This module provides a standardized configuration system for the Ember framework. It's designed to be simple to use while being flexible enough to handle complex configurations.

## Key Features

- **Clean API**: Simple, consistent interface with clear entry points
- **Thread Safety**: Thread-safe configuration access and updates
- **Multiple Sources**: Load from files and environment variables
- **Environment Resolution**: Replace `${VAR}` patterns with environment values
- **Validation**: Type checking and validation with Pydantic
- **Extensibility**: All schema classes support arbitrary additional fields

## Basic Usage

```python
from ember.core.config import load_config, create_config_manager

# Option 1: Load configuration directly
config = load_config()

# Access configuration
if config.registry.auto_discover:
    print("Auto-discovery is enabled")
    
# Option 2: Use ConfigManager for more features
config_manager = create_config_manager()
config = config_manager.get_config()

# Update configuration
config_manager.set_provider_api_key("openai", "sk-your-key")
```

## Configuration Sources

Configuration is loaded from the following sources, in order (later sources override earlier ones):

1. Default values defined in schema classes
2. YAML file (`config.yaml` in current directory by default)
3. Environment variables with `EMBER_` prefix

### Environment Variables

Environment variables are mapped to configuration keys:

```
EMBER_REGISTRY_AUTO_DISCOVER=true → config.registry.auto_discover = True
EMBER_LOGGING_LEVEL=DEBUG → config.logging.level = "DEBUG"
```

You can also reference environment variables within your YAML file:

```yaml
registry:
  providers:
    openai:
      api_keys:
        default:
          key: "${OPENAI_API_KEY}"
```

## Configuration Schema

The configuration schema is defined in a hierarchy of Pydantic models:

- `EmberConfig`: Top-level configuration
  - `registry`: Registry configuration
    - `providers`: Dictionary of providers
      - `<provider_name>`: Provider configuration
        - `models`: Dictionary of models
          - `<model_name>`: Model configuration
  - `logging`: Logging configuration

Each level allows arbitrary additional fields, so you can extend the configuration as needed.

## Working with Models and Providers

The configuration system provides helper methods for working with models and providers:

```python
# Get provider by name
provider = config.get_provider("openai")
if provider and provider.enabled:
    print(f"OpenAI API key: {provider.api_keys['default'].key}")
    
# Get model by ID
model = config.get_model_config("openai:gpt-4")
if model:
    cost = model.cost.calculate(100, 200)  # Calculate cost for tokens
    print(f"Cost: ${cost:.4f}")
```

## Example Configuration

See `config.yaml.example` for a complete example configuration file.

## Advanced Usage

### Custom Configuration Path

```python
# Load from a specific file
config = load_config(config_path="/path/to/custom-config.yaml")

# Or with the config manager
config_manager = create_config_manager(config_path="/path/to/custom-config.yaml")
```

### Custom Environment Prefix

```python
# Use MY_APP_ prefix instead of EMBER_
config = load_config(env_prefix="MY_APP")
```

### Thread-Safe Updates

```python
# Update configuration in a thread-safe manner
config_manager = create_config_manager()
config_manager.set_provider_api_key("openai", "new-key")

# Get the updated configuration
updated_config = config_manager.get_config()
```

### Schema Extensions

All schema classes allow arbitrary additional fields:

```yaml
# Standard fields
registry:
  auto_discover: true
  
  # Extended fields
  cache_ttl: 3600
  experimental_features: true
```

You can access these extended fields directly:

```python
ttl = config.registry.cache_ttl  # 3600
```