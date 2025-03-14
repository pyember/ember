# Ember Configuration System

The Ember configuration system provides a robust, extensible, and thread-safe way to manage application settings. It supports multiple configuration sources, schema validation, environment variable resolution, and a comprehensive transformation pipeline.

## Core Concepts

The configuration system is built around these key components:

1. **ConfigManager**: Orchestrates loading, validating, and providing access to configuration
2. **Schema Models**: Pydantic models that define the structure and validation rules

## Getting Started

### Basic Usage

The simplest way to use the configuration system is with the config manager:

```python
from ember.core.config.manager import create_config_manager

# Create a configuration manager
config_manager = create_config_manager()

# Access configuration
config = config_manager.get_config()
auto_discover = config.registry.auto_discover
openai_enabled = config.registry.providers["openai"].enabled
```

### Configuration Sources

By default, Ember looks for configuration in these locations, in order:

1. Custom path provided to `create_config_manager(config_path="path/to/config.yaml")`
2. Path specified in the `EMBER_CONFIG` environment variable
3. `./config.yaml` in the current working directory

### Environment Variables

You can configure Ember using environment variables with the `EMBER_` prefix:

```bash
# Set configuration via environment variables
export EMBER_REGISTRY_AUTO_DISCOVER=false
export EMBER_LOGGING_LEVEL=DEBUG
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-...
```

### Using Environment Variables in YAML

You can reference environment variables in your YAML configuration:

```yaml
registry:
  providers:
    openai:
      enabled: true
      api_keys:
        default:
          key: "${OPENAI_API_KEY}"
    anthropic:
      enabled: true
      api_keys:
        default:
          key: "${ANTHROPIC_API_KEY}"
```

## Configuration Schema

The default Ember configuration schema includes:

- **registry**: Model registry settings
  - **auto_discover**: Whether to auto-discover models from providers
  - **auto_register**: Whether to auto-register discovered models
  - **providers**: Provider-specific settings
    - **enabled**: Whether the provider is enabled
    - **api_keys**: API keys for the provider
    - **models**: Model-specific settings
- **logging**: Logging settings
  - **level**: Logging level
  - **format**: Log format string
- **data_paths**: Data paths settings
  - **datasets**: Path to datasets
  - **cache**: Path to cache

For a complete example, see `src/ember/core/registry/model/config/ember.yaml.example`.

## Application Context Integration

Ember's application context integrates with the configuration system:

```python
from ember.core.app_context import create_ember_app, get_ember_context

# Create app context with custom config
app_context = create_ember_app(config_path="my_config.yaml")

# Get config from app context
context = get_ember_context()
config = context.config_manager.get_config()
```

## Advanced Usage

### Custom Configuration Schema

You can define your own configuration schema:

```python
from pydantic import BaseModel
from ember.core.configs import ConfigManager, YamlFileProvider

class MyConfig(BaseModel):
    name: str
    version: str
    debug: bool = False

# Create manager with custom schema
config_manager = ConfigManager(
    schema_class=MyConfig,
    providers=[YamlFileProvider("my_config.yaml")]
)

config = config_manager.load()
print(f"App: {config.name} v{config.version}, Debug: {config.debug}")
```

### Custom Configuration Providers

You can create custom providers for different configuration sources:

```python
from typing import Dict, Any
from ember.core.configs import ConfigProvider, ConfigManager

class DatabaseProvider(ConfigProvider):
    def __init__(self, connection_string):
        self.connection_string = connection_string
    
    def load(self) -> Dict[str, Any]:
        # Load configuration from database
        # Implementation depends on your database
        ...
        
    def save(self, config: Dict[str, Any]) -> None:
        # Save configuration to database
        ...

# Use the custom provider
config_manager = ConfigManager(
    schema_class=MyConfig,
    providers=[DatabaseProvider("mysql://user:pass@localhost/db")]
)
```

### Custom Transformers

You can create custom transformations:

```python
from ember.core.configs import ConfigTransformer

# Define a transformation function
def uppercase_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    """Transform all string values to uppercase."""
    result = {}
    for key, value in config.items():
        if isinstance(value, str):
            result[key] = value.upper()
        elif isinstance(value, dict):
            result[key] = uppercase_keys(value)
        else:
            result[key] = value
    return result

# Create a transformer
transformer = ConfigTransformer()
transformer.add_transformation(uppercase_keys)

# Use the transformer
config_manager = ConfigManager(
    schema_class=MyConfig,
    providers=[YamlFileProvider("config.yaml")],
    transformers=[transformer]
)
```

### Multiple Configuration Files

You can use multiple configuration files that override each other:

```python
from ember.core.configs import ConfigManager, YamlFileProvider

# Create providers (later providers override earlier ones)
providers = [
    YamlFileProvider("base.yaml"),     # Base configuration
    YamlFileProvider("dev.yaml")       # Development overrides
]

# Create manager with multiple providers
config_manager = ConfigManager(schema_class=MyConfig, providers=providers)
```

### Dynamically Updating Configuration

You can update configuration at runtime:

```python
# Update a provider API key
config_manager.set_provider_api_key("openai", "new_api_key")

# Save the updated configuration
config_manager.save()

# Reload configuration
config_manager.reload()
```

## Thread Safety

The configuration system is designed to be thread-safe:

- All ConfigManager methods use a reentrant lock
- Multiple threads can safely read configuration concurrently
- Writes are properly synchronized
- Configuration changes are atomic

## Configuration Architecture

The configuration system follows these SOLID principles:

1. **Single Responsibility Principle**: Each component has a focused responsibility:
   - ConfigManager orchestrates the configuration lifecycle
   - Providers load data from specific sources
   - Transformers apply specific transformations
   - Schema models handle validation

2. **Open/Closed Principle**: The system is open for extension:
   - Create custom providers for new sources
   - Add custom transformers for new transformations
   - Extend schema with additional fields

3. **Liskov Substitution Principle**: Implementations are interchangeable:
   - Any ConfigProvider can be used where a provider is expected
   - Any transformation function can be used in a ConfigTransformer

4. **Interface Segregation Principle**: Interfaces are focused and minimal:
   - Providers need only implement `load()` (and optionally `save()`)
   - Transformers use simple function interfaces

5. **Dependency Inversion Principle**: High-level modules depend on abstractions:
   - ConfigManager depends on the ConfigProvider interface, not implementations
   - ConfigManager depends on the transformation function protocol, not details

## Transitioning from Legacy Configuration

Ember maintains compatibility with code using the older configuration system:

- The application context handles the translation layer
- API keys set in the new system are propagated to the old system
- Configuration values from the new system can be accessed through the old API

## Examples

For complete configuration examples, see:

- [Configuration Example](https://github.com/ember-ai/ember/blob/main/src/ember/examples/configuration_example.py)
- [Model Registry Example](https://github.com/ember-ai/ember/blob/main/src/ember/examples/model_registry_example.py)

## Best Practices

1. **Use Environment Variables for Secrets**:
   ```yaml
   api_keys:
     default:
       key: "${OPENAI_API_KEY}"
   ```

2. **Layer Your Configuration**:
   - base.yaml: Default settings for all environments
   - dev.yaml: Development environment overrides
   - prod.yaml: Production environment overrides

3. **Validate Configuration Early**:
   ```python
   # Load and validate on startup
   config = config_manager.load()
   ```

4. **Use Type Hints**:
   ```python
   # Get strongly-typed configuration
   config: EmberConfig = config_manager.get_config()
   ```

5. **Create Environment-Specific Config Providers**:
   ```python
   # Development
   config_manager = create_default_config_manager("config.dev.yaml")
   
   # Production
   config_manager = create_default_config_manager("config.prod.yaml")
   ```