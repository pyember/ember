# Ember Model Module - Quickstart Guide

This quickstart guide will help you integrate LLM models into your project using Ember's model module. The guide follows SOLID principles and best practices for modularity and maintainability.

## 1. Installation

```bash
pip install ember-ai
```

## 2. API Key Setup

Set your API keys as environment variables:

```bash
# For bash/zsh
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"

# For Windows PowerShell
$env:OPENAI_API_KEY="your-openai-key"
$env:ANTHROPIC_API_KEY="your-anthropic-key"
$env:GOOGLE_API_KEY="your-google-key"
```

Alternatively, create a `.env` file in your project:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
```

## 3. Basic Usage (One Line)

```python
import ember

# Initialize and use in one line
service = ember.init()
response = service("openai:gpt-4o", "Hello world!")
print(response.data)
```

## 4. Standard Usage Pattern

```python
from ember import initialize_ember
from ember.core.registry.model.base.services.model_service import ModelService

# Initialize the model registry
registry = initialize_ember(auto_register=True)

# Create a service instance
service = ModelService(registry=registry)

# Use the service to invoke models
response = service.invoke_model(
    model_id="anthropic:claude-3.5-sonnet", 
    prompt="Explain quantum computing in 50 words"
)
print(response.data)
```

## 5. Direct Model Access (Pytorch-like)

```python
# Get model directly for more control
model = service.get_model("openai:gpt-4o")

# Use the model directly
response = model("What's the capital of France?")
print(response.data)
```

## 6. Usage Tracking

```python
from ember import initialize_ember
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.base.services.usage_service import UsageService

# Initialize with usage tracking
registry = initialize_ember(auto_register=True)
usage_service = UsageService()
service = ModelService(registry=registry, usage_service=usage_service)

# Make model requests
response = service("openai:gpt-4o", "Hello world!")

# Get usage statistics
total_usage = usage_service.get_total_usage()
print(f"Total tokens: {total_usage.tokens}")
print(f"Total cost: ${total_usage.cost}")
```

## 7. Available Models

You can use any of these models by their ID:

- `openai:gpt-4o`
- `openai:gpt-4o-mini`
- `openai:o1`
- `anthropic:claude-3.5-sonnet`
- `deepmind:gemini-1.5-pro`

## 8. Error Handling

```python
try:
    response = service("openai:gpt-4o", "Hello world!")
    print(response.data)
except Exception as e:
    print(f"Error: {str(e)}")
```

## 9. Advanced: Adding Custom Models

```python
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.schemas.model_info import ModelInfo

# Create model info
custom_model = ModelInfo(
    model_id="custom:my-model",
    model_name="my-custom-model",
    provider={
        "name": "CustomProvider",
        "default_api_key": "${CUSTOM_API_KEY}"
    }
)

# Register custom model
registry = ModelRegistry()
registry.register_model(custom_model)
```

## 10. Type-safe Model Invocation with Enums

```python
from ember import initialize_ember
from ember.core.registry.model.config.model_enum import ModelEnum

service = initialize_ember(auto_register=True)

# Use enum for type-safety
response = service(ModelEnum.OPENAI_GPT4O, "Hello world!")
print(response.data)
```

## Next Steps

For more advanced usage, check out:
- Custom provider integration
- Multi-model ensembling
- Prompt templating
- Streaming responses