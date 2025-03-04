"""
Model Registry Usage Example - Demonstrates patterns for integrating LLMs.

This example shows:
1. The one-line initialization pattern 
2. The standard initialization pattern
3. Direct model access (PyTorch-like pattern)
4. Usage tracking and cost estimation
5. Batch processing with multiple models
6. Working with model enums for type safety
7. Adding custom models to the registry

For comprehensive documentation, see:
docs/quickstart/model_registry.md
"""

import os
import logging
from typing import Dict, Any, List
import time
from concurrent.futures import ThreadPoolExecutor

from ember import initialize_ember
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.base.services.usage_service import UsageService
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.schemas.model_info import ModelInfo, ModelCost, RateLimit
from ember.core.registry.model.base.schemas.usage import UsageStats
from ember.core.registry.model.config.model_enum import ModelEnum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def one_line_pattern():
    """Demonstrates the simplest one-line initialization pattern."""
    # Import the init function
    # In a real project, this would be: from ember import init
    from ember.core.registry.model.config.settings import initialize_ember as init
    
    # Initialize and get a ready-to-use service in one line
    service = init(usage_tracking=True)
    
    # Use the service directly with a model ID
    try:
        response = service("openai:gpt-4o", "What is the capital of France?")
        print("\n=== One-line pattern result ===")
        print(response.data)
        return service  # Return for reuse in other examples
    except Exception as e:
        logger.exception(f"Error in one-line pattern: {e}")
        return None


def standard_pattern():
    """Demonstrates the standard initialization pattern with more control."""
    try:
        # Initialize the registry
        registry = initialize_ember(auto_register=True)
        
        # Create a service instance with usage tracking
        usage_service = UsageService()
        service = ModelService(registry=registry, usage_service=usage_service)
        
        # Use the service to invoke a model
        response = service.invoke_model(
            model_id="anthropic:claude-3.5-sonnet",
            prompt="Explain quantum computing in one sentence."
        )
        
        print("\n=== Standard pattern result ===")
        print(f"Response: {response.data}")
        
        # Check usage statistics
        usage = usage_service.get_total_usage()
        print(f"Total tokens used: {usage.tokens}")
        print(f"Estimated cost: ${usage.cost:.6f}")
        
        return service, usage_service
    except Exception as e:
        logger.exception(f"Error in standard pattern: {e}")
        return None, None


def direct_model_pattern():
    """Demonstrates direct model access (PyTorch-like pattern)."""
    try:
        # Initialize the registry
        registry = initialize_ember(auto_register=True)
        service = ModelService(registry=registry)
        
        # Get the model directly
        model = service.get_model("openai:gpt-4o")
        
        # Use the model directly, like you would use a PyTorch module
        response = model("What is the tallest mountain in the world?")
        
        print("\n=== Direct model pattern result ===")
        print(response.data)
        
        return service
    except Exception as e:
        logger.exception(f"Error in direct model pattern: {e}")
        return None


def type_safe_enum_pattern():
    """Demonstrates using ModelEnum for type-safe model references."""
    try:
        # Initialize with the standard pattern
        registry = initialize_ember(auto_register=True)
        service = ModelService(registry=registry)
        
        # Use enum instead of string literals
        response = service.invoke_model(
            model_id=ModelEnum.OPENAI_GPT4O,  # Type-safe enum
            prompt="What's your favorite programming language and why?"
        )
        
        print("\n=== Type-safe enum pattern result ===")
        print(response.data)
        
        # Access model metadata via enum
        model_info = service.registry.get_model_info(ModelEnum.OPENAI_GPT4O)
        print("\nModel metadata:")
        print(f"Name: {model_info.name}")
        print(f"Provider: {model_info.provider['name']}")
        print(f"Input cost per 1K tokens: ${model_info.cost.input_cost_per_thousand}")
        print(f"Output cost per 1K tokens: ${model_info.cost.output_cost_per_thousand}")
    except Exception as e:
        logger.exception(f"Error in type-safe enum pattern: {e}")


def batch_processing_pattern():
    """Demonstrates batch processing with multiple models."""
    try:
        # Initialize
        registry = initialize_ember(auto_register=True)
        usage_service = UsageService()
        service = ModelService(registry=registry, usage_service=usage_service)
        
        # Define prompts and models
        prompts = [
            "What is machine learning?",
            "Explain the concept of a neural network.",
            "What is transfer learning?",
            "Describe reinforcement learning."
        ]
        
        models = [
            ModelEnum.OPENAI_GPT4O,
            ModelEnum.OPENAI_GPT4O_MINI,
            ModelEnum.ANTHROPIC_CLAUDE_3_SONNET,
            ModelEnum.ANTHROPIC_CLAUDE_3_HAIKU
        ]
        
        # Process in parallel
        def process_prompt(args):
            model_id, prompt = args
            try:
                start_time = time.time()
                response = service.invoke_model(model_id=model_id, prompt=prompt)
                duration = time.time() - start_time
                return model_id, prompt, response.data, duration
            except Exception as e:
                return model_id, prompt, f"Error: {str(e)}", 0
        
        print("\n=== Batch processing results ===")
        tasks = list(zip(models, prompts))
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_prompt, tasks))
        
        # Print results
        for i, (model, prompt, result, duration) in enumerate(results):
            print(f"\nTask {i+1}:")
            print(f"Model: {model}")
            print(f"Prompt: {prompt}")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Result: {result[:100]}..." if len(result) > 100 else f"Result: {result}")
        
        # Show aggregated usage
        usage = usage_service.get_total_usage()
        print("\nTotal usage across all batch operations:")
        print(f"Total tokens: {usage.tokens}")
        print(f"Estimated cost: ${usage.cost:.6f}")
    except Exception as e:
        logger.exception(f"Error in batch processing pattern: {e}")


def custom_model_pattern():
    """Demonstrates adding custom models to the registry."""
    try:
        # Create a new registry (don't use the global one)
        registry = ModelRegistry()
        
        # Register a custom model
        custom_model = ModelInfo(
            id="custom:my-llm",
            name="My Custom LLM",
            cost=ModelCost(
                input_cost_per_thousand=3.0,
                output_cost_per_thousand=6.0
            ),
            rate_limit=RateLimit(
                tokens_per_minute=50000,
                requests_per_minute=1000
            ),
            provider={
                "name": "CustomProvider",
                "default_api_key": "${CUSTOM_API_KEY}"
            }
        )
        
        registry.register_model(custom_model)
        
        # List all models
        model_ids = registry.list_models()
        
        print("\n=== Custom model registration ===")
        print(f"Registered models: {model_ids}")
        
        # Check model exists and get info
        exists = registry.model_exists("custom:my-llm")
        if exists:
            info = registry.get_model_info("custom:my-llm")
            print("\nCustom model details:")
            print(f"ID: {info.id}")
            print(f"Name: {info.name}")
            print(f"Provider: {info.provider['name']}")
            print(f"Input cost: ${info.cost.input_cost_per_thousand} per 1K tokens")
        else:
            print("Custom model registration failed!")
    except Exception as e:
        logger.exception(f"Error in custom model pattern: {e}")


def main():
    """Run all example patterns."""
    print("Running Model Registry examples...\n")
    print("Make sure you have set up your API keys in environment variables:")
    print("  - OPENAI_API_KEY")
    print("  - ANTHROPIC_API_KEY")
    print("  - GOOGLE_API_KEY (if using Gemini models)")
    
    # Run each pattern
    service = one_line_pattern()
    service, usage_service = standard_pattern()
    direct_model_pattern()
    type_safe_enum_pattern()
    custom_model_pattern()
    
    # Only run batch processing if we have API keys configured
    keys_needed = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing_keys = [key for key in keys_needed if os.environ.get(key) is None]
    if not missing_keys:
        batch_processing_pattern()
    else:
        print("\nSkipping batch processing example due to missing API keys.")
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()