# Ember Configuration System

The Ember configuration system provides a thread-safe way to manage application settings with schema validation and environment variable resolution.

## Core Components

1. **ConfigManager**: Central access point for configuration
2. **EmberConfig**: Pydantic schema defining configuration structure 
3. **Loader**: Handles loading and merging configuration from sources

## Basic Usage

```python
from ember.core.config.manager import create_config_manager

# Create a configuration manager
config_manager = create_config_manager()

# Access configuration
config = config_manager.get_config()
auto_discover = config.registry.auto_discover  # or config.model_registry for legacy code
openai_enabled = config.get_provider("openai").enabled
```

## Configuration Sources

Ember looks for configuration in these locations, in order:

1. Custom path provided to `create_config_manager(config_path="path/to/config.yaml")`
2. Path specified in the `EMBER_CONFIG` environment variable
3. `./config.yaml` in the current working directory

## Environment Variables

Configure Ember using environment variables with the `EMBER_` prefix:

```bash
# Set configuration via environment variables
export EMBER_REGISTRY_AUTO_DISCOVER=false
export EMBER_LOGGING_LEVEL=DEBUG
```

Common API keys are automatically loaded from environment:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-...
```

You can reference environment variables in YAML configuration:

```yaml
registry:
  providers:
    openai:
      enabled: true
      api_keys:
        default:
          key: "${OPENAI_API_KEY}"
```

## Configuration Schema

The configuration schema includes:

- **registry**: Model registry settings (legacy code may use `model_registry`)
  - **auto_discover**: Whether to auto-discover models
  - **auto_register**: Whether to auto-register discovered models
  - **providers**: Provider-specific settings
    - **enabled**: Whether the provider is enabled
    - **api_keys**: API keys for the provider
    - **models**: Model-specific settings
      - **id**: Unique model identifier
      - **name**: Display name for the model
      - **cost_input**: Cost per 1000 input tokens
      - **cost_output**: Cost per 1000 output tokens
      - **rate_limit**: Rate limiting configuration
- **logging**: Logging settings
  - **level**: Logging level

Example configuration:

```yaml
registry:
  auto_discover: true
  providers:
    openai:
      enabled: true
      api_keys:
        default:
          key: "${OPENAI_API_KEY}"
      models:
        gpt-4:
          id: "gpt-4"
          name: "GPT-4"
          cost_input: 5.0
          cost_output: 15.0
```

## Application Context Integration

Ember's application context integrates with the configuration system:

```python
from ember.core.app_context import create_ember_app, get_ember_context

# Create app context with custom config
app_context = create_ember_app(config_path="my_config.yaml")

# Get config from app context
context = get_ember_context()
config = context.config_manager.get_config()

# Access configuration 
auto_discover = config.registry.auto_discover
openai_config = config.get_provider("openai")

# API keys from environment variables are automatically loaded
model_registry = context.model_registry
openai_model = model_registry.get_model_info("openai:gpt-4")
```

## API Methods

Key ConfigManager methods:

```python
# Create a configuration manager
config_manager = create_config_manager(config_path="config.yaml")

# Get configuration
config = config_manager.get_config()

# Set a provider API key
config_manager.set_provider_api_key("openai", "new-api-key")

# Reload configuration from sources
config = config_manager.reload()
```

Key EmberConfig methods:

```python
# Get provider configuration
provider = config.get_provider("openai")

# Get model configuration
model = config.get_model_config("openai:gpt-4")

# Access model cost information (both forms work)
cost_input = model.cost_input  # Direct field access
cost_per_thousand = model.cost.input_cost_per_thousand  # Via computed cost property
```

## Model Registry Integration

Initialize the model registry with configuration:

```python
from ember.core.registry.model.initialization import initialize_registry

# Initialize with configuration
registry = initialize_registry(config_path="config.yaml")

# Or use an existing config manager
registry = initialize_registry(config_manager=config_manager)
```

## Backward Compatibility

For legacy code compatibility:

```python
# Old way - still works but shows deprecation warning
from ember.core.registry.model.config.settings import initialize_ember
registry = initialize_ember(auto_discover=True)

# New way - preferred approach
from ember.core.registry.model.initialization import initialize_registry
registry = initialize_registry()
```

## Best Practices

1. **Store secrets in environment variables**:
   ```yaml
   registry:
     providers:
       openai:
         api_keys:
           default:
             key: "${OPENAI_API_KEY}"
   ```

2. **Validate configuration early**:
   ```python
   config_manager = create_config_manager()
   config = config_manager.get_config()  # Validates on first access
   ```

3. **Create environment-specific configs**:
   ```python
   # Development
   config_manager = create_config_manager("config.dev.yaml")
   
   # Production
   config_manager = create_config_manager("config.prod.yaml")
   ```