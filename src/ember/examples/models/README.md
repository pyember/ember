# Ember Model Examples

This directory contains examples demonstrating how to use Ember's model registry system to work with various LLM providers.

## Examples

- `model_registry_example.py` - General usage patterns for the model registry
- `list_models.py` - How to list available models and their capabilities
- `model_registry_direct.py` - Direct usage of the model registry API
- `model_api_example.py` - Using the model API for inference
- `manual_model_registration.py` - Manually registering custom models
- `register_models_directly.py` - Direct model registration example

## Best Practices

### 1. Accessing the Registry

```python
from ember.api import models

# Get the registry
registry = models.get_registry()
```

### 2. Using Provider Namespaces

```python
from ember.api import models

# Direct invocation with namespace
response = models.openai.gpt4o("What is the capital of France?")
response = models.anthropic.claude("Tell me about quantum physics")
```

### 3. Getting and Using Model Instances

```python
from ember.api import models

# Get model service
model_service = models.get_model_service()

# Get a model instance
model = model_service.get_model("openai:gpt-4o")

# Use the model instance
response = model(prompt="What is the capital of France?")
print(response.data)

# Alternative: direct invocation
response = model_service.invoke_model("openai:gpt-4o", "What is the capital of France?")
```

### 4. Model Registration with ProviderInfo

```python
from ember.api.models import ModelInfo, ProviderInfo, ModelCost, RateLimit

model_info = ModelInfo(
    id="provider:model-name",
    name="Human-Readable Name",
    context_window=32000,
    cost=ModelCost(
        input_cost_per_thousand=0.001,
        output_cost_per_thousand=0.002,
    ),
    provider=ProviderInfo(
        name="Provider Name",
        default_api_key="YOUR_API_KEY",
        base_url="https://api.provider.com",
    ),
)

# Register the model
registry = models.get_registry()
registry.register_model(model_info=model_info)
```

## Running Examples

To run any example, use the following command format:

```bash
# Using uv (recommended)
uv run python src/ember/examples/models/example_name.py

# Or if in an activated virtual environment
python src/ember/examples/models/example_name.py
```

Replace `example_name.py` with the desired example file.

## API Keys

Most examples require API keys for LLM providers to be set in your environment:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Next Steps

After understanding model usage, explore:

- `operators/` - For examples of building computation with models
- `advanced/` - For complex model usage patterns
