# Ember Configuration System

This directory contains Ember's unified configuration system, which provides a flexible, type-safe, and extensible way to configure the framework. The configuration system is now the central source of truth for all Ember configuration, including model registry settings.

## Key Features

- **Multiple configuration sources**: Configuration can come from YAML files, environment variables, and code
- **Schema validation**: Pydantic models ensure type safety and validation
- **Environment variable resolution**: Replace values with environment variables using the `${VAR_NAME}` syntax
- **Thread safety**: All operations are thread-safe for concurrent access
- **Extensibility**: Easy to add new configuration sources and transformations

## Using the Configuration System

### Basic Usage

```python
from ember.core.configs import create_default_config_manager

# Create a default configuration manager (loads from ./config.yaml or EMBER_CONFIG env var)
config_manager = create_default_config_manager()

# Load configuration
config = config_manager.get_config()

# Access configuration values
auto_discover = config.model_registry.auto_discover
logging_level = config.logging.level
```

### Custom Configuration File

```python
from ember.core.configs import create_default_config_manager

# Specify a custom configuration file
config_manager = create_default_config_manager(config_path="/path/to/config.yaml")
config = config_manager.get_config()
```

### Environment Variables

The configuration system automatically loads environment variables with the `EMBER_` prefix:

```
# These environment variables are automatically mapped to configuration settings
EMBER_MODEL_REGISTRY_AUTO_DISCOVER=true
EMBER_LOGGING_LEVEL=DEBUG
EMBER_OPENAI_API_KEY=sk-...
```

You can also reference environment variables in your YAML config using the `${VAR_NAME}` syntax:

```yaml
model_registry:
  providers:
    openai:
      api_keys:
        default:
          key: "${OPENAI_API_KEY}"
```

## Configuration Schema

The configuration schema is defined in `schema.py` using Pydantic models:

- `EmberConfig`: Top-level configuration
- `ModelRegistryConfig`: Model registry settings
- `ProviderConfig`: Provider-specific settings
- `ModelConfig`: Model-specific settings
- `ApiKeyConfig`: API key configuration
- `LoggingConfig`: Logging settings
- `DataPathsConfig`: Data paths settings

See the example configuration file at `config.yaml.example` in the project root directory for a complete example.

## Advanced Usage

### Custom Configuration Schema

```python
from ember.core.configs import ConfigManager
from pydantic import BaseModel, Field

# Define a custom configuration schema
class MyAppConfig(BaseModel):
    name: str
    version: str
    debug: bool = False

# Create a configuration manager with the custom schema
config_manager = ConfigManager(
    schema_class=MyAppConfig,
    providers=[YamlFileProvider("myapp.yaml")],
    transformers=[ConfigTransformer([resolve_env_vars])]
)

# Load and access configuration
config = config_manager.load()
print(f"App: {config.name} v{config.version}")
```

### Custom Configuration Providers

```python
from ember.core.configs import ConfigProvider, ConfigManager

# Define a custom configuration provider
class DatabaseConfigProvider(ConfigProvider[dict]):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def load(self) -> dict:
        # Load configuration from database
        # (implementation depends on your database)
        ...
        
    def save(self, config: dict) -> None:
        # Save configuration to database
        ...

# Use the custom provider
config_manager = ConfigManager(
    schema_class=EmberConfig,
    providers=[
        YamlFileProvider("config.yaml"),
        DatabaseConfigProvider("mysql://user:pass@localhost/db")
    ]
)
```

## Configuration File Locations

The system looks for configuration files in the following order:

1. Custom path provided to `create_default_config_manager(config_path=...)`
2. Path specified in the `EMBER_CONFIG` environment variable
3. `./config.yaml` in the current working directory

## Default Values

Default values are defined in `schema.py` in the `DEFAULT_CONFIG` dictionary and in the Pydantic model field defaults.

## Integration with Model Registry

The model registry is fully integrated with the centralized configuration system. The initialization flow is:

1. Configuration is loaded through `create_default_config_manager()`
2. API keys are injected from environment variables if available
3. Model registry is initialized using `initialize_registry()` in `src/ember/core/registry/model/initialization.py`
4. Models are registered from configuration, with proper conversion from configuration schema to registry schema
5. Provider discovery runs if auto-discovery is enabled

This integration eliminates duplication and ensures a single source of truth for all configuration.

### Provider Integration

All model providers have been updated to use the centralized configuration system:

- **Anthropic**: Uses `get_provider("anthropic")` to retrieve API keys and settings
- **OpenAI**: Uses `get_provider("openai")` to retrieve API keys and settings
- **Google/DeepMind**: Uses `get_provider("google")` to retrieve API keys and settings

Each provider follows a consistent pattern:
1. First check the centralized configuration for API keys and settings
2. Fall back to environment variables if needed
3. Use default values as a last resort

### Model Registry Bridge

The `initialization.py` module serves as a bridge between the configuration system and the model registry. It handles:

- Converting configuration models to registry models
- Processing provider-specific settings
- Extracting and formatting cost information
- Managing API keys securely
- Handling errors gracefully with detailed logging

### Backward Compatibility

For backward compatibility, the old configuration system is still accessible through:

```python
from ember.core.registry.model.config.settings import initialize_ember
```

This function is now a thin wrapper around the new `initialize_registry()` function, which delegates to the centralized configuration system. It also issues a deprecation warning to encourage migration to the new API.

### Configuration Example

For a complete example of model registry configuration, see the `config.yaml.example` file in the project root directory, which demonstrates:

- API key configuration with environment variables
- Model registration with costs and rate limits
- Provider-specific settings
- Logging configuration