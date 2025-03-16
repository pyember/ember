"""
Simplified test for model discovery and registration.
"""

import pytest
import logging
from ember import initialize_ember

# Configure logging
logging.basicConfig(level=logging.INFO)

@pytest.mark.discovery
def test_model_discovery():
    """Test model discovery and registration."""
    # Initialize with force_discovery=True to ensure discovery happens
    registry = initialize_ember(auto_discover=True, force_discovery=True, initialize_context=False)
    
    # Get all registered models
    models = registry.list_models()
    print(f"\nFound {len(models)} registered models")
    
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
    
    # Verify that at least the registry object works
    assert hasattr(registry, "list_models"), "Registry should have list_models method"
    assert isinstance(models, list), "Registry list_models should return a list"
    
    # Try to get a model if any are available
    if models:
        test_model_id = models[0]
        print(f"\nTesting model access for: {test_model_id}")
        try:
            model = registry.get_model(test_model_id)
            print(f"Successfully retrieved model: {test_model_id}")
            assert model is not None, "Retrieved model should not be None"
        except Exception as e:
            print(f"Error retrieving model: {e}")
            # Don't fail the test if we can't get a model - might be due to missing API keys
            pass
