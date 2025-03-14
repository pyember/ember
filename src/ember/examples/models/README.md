# Ember Model Examples

This directory contains examples demonstrating how to use Ember's model registry system to work with various LLM providers.

## Examples

- `model_registry_example.py` - General usage patterns for the model registry
- `list_models.py` - How to list available models and their capabilities
- `model_registry_direct.py` - Direct usage of the model registry API
- `model_api_example.py` - Using the model API for inference
- `manual_model_registration.py` - Manually registering custom models
- `register_models_directly.py` - Direct model registration example

## Running Examples

To run any example, use the following command format:

```bash
poetry run python src/ember/examples/models/example_name.py
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
