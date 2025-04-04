# Ember Model Module - Quickstart Guide

This quickstart guide will help you integrate LLM models into your project using Ember's model module. The guide follows SOLID principles and best practices for modularity and maintainability.

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/pyember/ember.git
cd ember

# Install using uv
uv pip install -e "."
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

## 3. Basic Usage (Provider Namespaces)

```python
from ember.api import models

# Access provider models directly via namespaces
response = models.openai.gpt4o("What is the capital of France?")
print(response.data)
```

## 4. Builder Pattern

```python
from ember.api.models import ModelBuilder

# Create model with builder pattern
model = (
    ModelBuilder()
    .temperature(0.7)
    .max_tokens(100)
    .build("anthropic:claude-3-5-sonnet")
)

# Generate response
response = model.generate(prompt="Explain quantum computing in 50 words")
print(response.data)
```

## 5. Direct Model Access (ModelAPI)

```python
from ember.api.models import ModelAPI

# Get model directly for more control
model = ModelAPI(id="openai:gpt-4o")

# Use the model directly
response = model.generate(prompt="What's the capital of France?")
print(response.data)
```

## 6. Usage Tracking

```python
from ember.api.models import get_usage_service

# Access usage service
usage_service = get_usage_service()

# Make model requests using any method
response = models.anthropic.claude_3_5_sonnet("Hello world!")

# Get usage statistics for a specific model
model_id = "anthropic:claude-3-5-sonnet"
usage_summary = usage_service.get_usage_summary(model_id=model_id)

print(f"Model: {usage_summary.model_name}")
print(f"Total tokens: {usage_summary.total_tokens_used}")
print(f"Prompt tokens: {usage_summary.total_usage.prompt_tokens}")
print(f"Completion tokens: {usage_summary.total_usage.completion_tokens}")
print(f"Estimated cost: ${usage_summary.total_usage.cost_usd:.4f}")
```

## 7. Available Models

You can use any of these models by their ID or corresponding ModelEnum:

### OpenAI Models
- `openai:gpt-4o` or `ModelEnum.gpt4o`
- `openai:gpt-4o-mini` or `ModelEnum.gpt4o_mini`
- `openai:gpt-4` or `ModelEnum.gpt4`
- `openai:gpt-4-turbo` or `ModelEnum.gpt4_turbo`
- `openai:gpt-3.5-turbo` or `ModelEnum.gpt3_5_turbo`
- `openai:o1-2024-12-17` or `ModelEnum.o1`

### Anthropic Models
- `anthropic:claude-3.7-sonnet` or `ModelEnum.claude_3_7_sonnet`
- `anthropic:claude-3-5-sonnet` or `ModelEnum.claude_3_5_sonnet`
- `anthropic:claude-3-5-haiku` or `ModelEnum.claude_3_5_haiku`
- `anthropic:claude-3-opus` or `ModelEnum.claude_3_opus`
- `anthropic:claude-3-haiku` or `ModelEnum.claude_3_haiku`

### Deepmind Models
- `deepmind:gemini-1.5-pro` or `ModelEnum.gemini_1_5_pro`
- `deepmind:gemini-1.5-flash` or `ModelEnum.gemini_1_5_flash`
- `deepmind:gemini-1.5-flash-8b` or `ModelEnum.gemini_1_5_flash_8b`
- `deepmind:gemini-2.0-flash` or `ModelEnum.gemini_2_0_flash`
- `deepmind:gemini-2.0-flash-lite` or `ModelEnum.gemini_2_0_flash_lite`
- `deepmind:gemini-2.0-pro` or `ModelEnum.gemini_2_0_pro`

## 8. Error Handling

```python
try:
    response = models.openai.gpt4o("Hello world!")
    print(response.data)
except Exception as e:
    print(f"Error: {str(e)}")
```

## 9. Advanced: Adding Custom Models

```python
from ember.api.models import get_registry, ModelInfo, ModelCost, RateLimit

# Get registry
registry = get_registry()

# Create model info
custom_model = ModelInfo(
    id="custom:my-model",
    name="my-custom-model",
    cost=ModelCost(
        input_cost_per_thousand=0.0005, 
        output_cost_per_thousand=0.0015
    ),
    rate_limit=RateLimit(
        tokens_per_minute=100000, 
        requests_per_minute=3000
    ),
    provider={
        "name": "CustomProvider",
        "default_api_key": "${CUSTOM_API_KEY}",
        "base_url": "https://api.custom-provider.com/v1"
    }
)

# Register custom model
registry.register_model(model_info=custom_model)
```

## 10. Type-safe Model Invocation with Enums

```python
from ember.api.models import ModelAPI, ModelEnum

# Create model using an enum for type safety
model = ModelAPI.from_enum(ModelEnum.gpt4o)

# Generate response
response = model.generate(prompt="Hello world!")
print(response.data)
```

## 11. Batch Processing with Multiple Models

```python
from concurrent.futures import ThreadPoolExecutor
from ember.api.models import ModelAPI

# Define prompts and models
prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "What is transfer learning?",
    "Describe reinforcement learning."
]

model_ids = [
    "openai:gpt-4o",
    "anthropic:claude-3-5-sonnet",
    "deepmind:gemini-1.5-pro",
    "openai:gpt-4-turbo"
]

# Process in parallel
def process_prompt(args):
    model_id, prompt = args
    model = ModelAPI(id=model_id)
    response = model.generate(prompt=prompt)
    return model_id, prompt, response.data

# Execute in parallel with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_prompt, zip(model_ids, prompts)))

# Print results
for model_id, prompt, result in results:
    print(f"Model: {model_id}")
    print(f"Prompt: {prompt}")
    print(f"Result: {result[:100]}...")  # Truncate for display
    print()
```

## Next Steps

For more advanced usage, check out:
- Custom provider integration
- Multi-model ensembling
- Prompt templating
- Streaming responses

## Provider Discovery System

The model registry integrates a robust provider discovery mechanism with three key design principles:

### 1. Resilience

The discovery system ensures application stability through:
- **Graceful degradation** - Fallback models when APIs are unreachable
- **Timeout protection** - Automatic 30s timeout with ThreadPoolExecutor to prevent indefinite blocking
- **Error isolation** - API failures in one provider don't affect others

### 2. Testing Strategy

Testing model discovery requires both:
- **Unit tests** - Fast, deterministic verification with mocks
- **Integration tests** - Selective API verification with real credentials

Enable integration tests with environment flags:
```bash
RUN_PROVIDER_INTEGRATION_TESTS=1 pytest tests/integration/core/registry/test_provider_discovery.py -v
```

### 3. Design Patterns

The discovery system employs:
- **Adapter Pattern** - Unified interface across varying provider APIs
- **Factory Pattern** - Dynamic provider instantiation based on available credentials
- **Dependency Injection** - Explicit API key configuration to avoid global state
- **Template Method** - Common discovery workflow with provider-specific implementations