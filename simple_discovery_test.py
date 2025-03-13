"""
Simplified test for model discovery and registration.
"""

import logging
from ember.core.registry.model.config.settings import initialize_ember

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize with force_discovery=True to ensure discovery happens
registry = initialize_ember(auto_discover=True, force_discovery=True)

# Print all registered models
print("\nRegistered models:")
models = registry.list_models()
print(f"Found {len(models)} registered models")

# Group by provider
openai_models = [model for model in models if model.startswith("openai:")]
anthropic_models = [model for model in models if model.startswith("anthropic:")]

print(f"OpenAI models: {len(openai_models)}")
print(f"Anthropic models: {len(anthropic_models)}")

# Print sample models
if openai_models:
    print("\nSample OpenAI models:")
    for model in openai_models[:3]:
        print(f"  - {model}")

if anthropic_models:
    print("\nSample Anthropic models:")
    for model in anthropic_models[:3]:
        print(f"  - {model}")

# Try to get a model
if models:
    test_model_id = models[0]
    print(f"\nTesting model access for: {test_model_id}")
    try:
        model = registry.get_model(test_model_id)
        print(f"Successfully retrieved model: {test_model_id}")
    except Exception as e:
        print(f"Error retrieving model: {e}")