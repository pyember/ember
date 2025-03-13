"""
Debug script for testing model discovery components directly.
"""

import logging
import os
from pprint import pprint

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import discovery components
from ember.core.registry.model.base.registry.discovery import ModelDiscoveryService
from ember.core.registry.model.providers.openai.openai_discovery import OpenAIDiscovery
from ember.core.registry.model.providers.anthropic.anthropic_discovery import AnthropicDiscovery

# 1. Test individual discovery providers
print("\n=== Testing Individual Discovery Providers ===\n")

# Test OpenAI discovery
print("Testing OpenAI discovery...")
openai_discovery = OpenAIDiscovery()
openai_models = openai_discovery.fetch_models()
print(f"Found {len(openai_models)} OpenAI models")
print("Sample OpenAI models:")
for i, (model_id, model_data) in enumerate(list(openai_models.items())[:3]):
    print(f"  {model_id}: {model_data.get('model_name')}")
    
# Test whether these are from API or fallbacks
print("\nChecking if OpenAI models are from API or fallbacks...")
fallback_models = openai_discovery._get_fallback_models()
api_discovered = any(model_id not in fallback_models for model_id in openai_models)
print(f"Contains non-fallback models: {api_discovered}")

# Test Anthropic discovery
print("\nTesting Anthropic discovery...")
anthropic_discovery = AnthropicDiscovery()
anthropic_models = anthropic_discovery.fetch_models()
print(f"Found {len(anthropic_models)} Anthropic models")
print("Sample Anthropic models:")
for i, (model_id, model_data) in enumerate(list(anthropic_models.items())[:3]):
    print(f"  {model_id}: {model_data.get('model_name')}")

# Test whether these are from API or fallbacks
print("\nChecking if Anthropic models are from API or fallbacks...")
fallback_models = anthropic_discovery._get_fallback_models()
api_discovered = any(model_id not in fallback_models for model_id in anthropic_models)
print(f"Contains non-fallback models: {api_discovered}")

# 2. Test the discovery service
print("\n=== Testing Model Discovery Service ===\n")
discovery_service = ModelDiscoveryService()
discovered_models = discovery_service.discover_models()
print(f"Found {len(discovered_models)} models via the discovery service")

# Print the first few models from each provider
print("\nSample discovered models:")
openai_ids = [model_id for model_id in discovered_models if model_id.startswith("openai:")]
anthropic_ids = [model_id for model_id in discovered_models if model_id.startswith("anthropic:")]

print(f"OpenAI models: {len(openai_ids)}")
print(f"Anthropic models: {len(anthropic_ids)}")

if openai_ids:
    print("\nOpenAI models (first 3):")
    for model_id in openai_ids[:3]:
        print(f"  - {model_id}")

if anthropic_ids:
    print("\nAnthropic models:")
    for model_id in anthropic_ids:
        print(f"  - {model_id}")