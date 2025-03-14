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

To run:
    poetry run python src/ember/examples/model_registry_example.py
"""

import os
import logging
from typing import Dict, Any, List
import time
from concurrent.futures import ThreadPoolExecutor

from ember.api import models
from ember.api.models import (
    ModelService, UsageService, ModelRegistry, 
    ModelInfo, ModelCost, RateLimit, UsageStats, ModelEnum
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def one_line_pattern():
    """Demonstrates the simplest one-line initialization pattern."""
    # With the new API, we don't need to initialize anything
    # Just use the models API directly
    
    try:
        # Direct model invocation
        response = models.openai.gpt4o("What is the capital of France?")
        print("\n=== One-line pattern result ===")
        print(response.text)
        return models  # Return for reuse in other examples
    except Exception as e:
        logger.exception(f"Error in one-line pattern: {e}")
        return None


def standard_pattern():
    """Demonstrates the standard initialization pattern with more control."""
    try:
        # With the new API, we can create a model instance with more control
        from ember.api.models import ModelBuilder
        
        # Create a model with specific parameters
        model = (
            ModelBuilder()
            .temperature(0.7)
            .max_tokens(100)
            .build("anthropic:claude-3.5-sonnet")
        )
        
        # Use the model
        response = model.generate(
            prompt="Explain quantum computing in one sentence."
        )
        
        print("\n=== Standard pattern result ===")
        print(f"Response: {response.text}")
        
        # Check usage statistics
        print(f"Total tokens used: {response.token_count}")
        print(f"Estimated cost: available through the models.usage API")
        
        return model
    except Exception as e:
        logger.exception(f"Error in standard pattern: {e}")
        return None


def direct_model_pattern():
    """Demonstrates direct model access (PyTorch-like pattern)."""
    try:
        # With the new API, we can use models directly
        
        # Use the direct model ID pattern
        response = models("openai:gpt-4o", "What is the tallest mountain in the world?")
        
        print("\n=== Direct model pattern result ===")
        print(response.text)
        
        return models
    except Exception as e:
        logger.exception(f"Error in direct model pattern: {e}")
        return None


def type_safe_enum_pattern():
    """Demonstrates using ModelEnum for type-safe model references."""
    try:
        # With the new API, we can use enums directly with models
        from ember.api.models import ModelAPI
        
        # Use enum instead of string literals
        model = ModelAPI.from_enum(ModelEnum.OPENAI_GPT4O)
        response = model.generate(
            prompt="What's your favorite programming language and why?"
        )
        
        print("\n=== Type-safe enum pattern result ===")
        print(f"Response: {response.text[:150]}...")  # Truncated for readability
        
        # Access model metadata
        model_info = models.registry.get_model_info(ModelEnum.OPENAI_GPT4O)
        print("\nModel metadata:")
        print(f"Name: {model_info.name}")
        print(f"Provider: {model_info.provider['name']}")
        print(f"Input cost per 1K tokens: ${model_info.cost.input_cost_per_thousand:.4f}")
        print(f"Output cost per 1K tokens: ${model_info.cost.output_cost_per_thousand:.4f}")
        print(f"Context window: {model_info.context_window} tokens")
        print(f"Version: {model_info.version if hasattr(model_info, 'version') else 'N/A'}")
    except Exception as e:
        logger.exception(f"Error in type-safe enum pattern: {e}")


def batch_processing_pattern():
    """Demonstrates batch processing with multiple models."""
    try:
        from ember.api.models import ModelAPI
        
        # Define prompts and models
        prompts = [
            "What is machine learning?",
            "Explain the concept of a neural network.",
            "What is transfer learning?",
            "Describe reinforcement learning."
        ]
        
        model_ids = [
            "openai:gpt-4o",
            "openai:gpt-4o-mini",
            "anthropic:claude-3-sonnet",
            "anthropic:claude-3-haiku"
        ]
        
        # Process in parallel
        def process_prompt(args):
            model_id, prompt = args
            try:
                model = ModelAPI.from_id(model_id)
                start_time = time.time()
                response = model.generate(prompt=prompt)
                duration = time.time() - start_time
                return model_id, prompt, response.text, duration
            except Exception as e:
                return model_id, prompt, f"Error: {str(e)}", 0
        
        print("\n=== Batch processing results ===")
        tasks = list(zip(model_ids, prompts))
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_prompt, tasks))
        
        # Print results
        for i, (model, prompt, result, duration) in enumerate(results):
            print(f"\nTask {i+1}:")
            print(f"Model: {model}")
            print(f"Prompt: {prompt}")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Result: {result[:100]}..." if len(result) > 100 else f"Result: {result}")
        
        # Calculate and show aggregate stats
        total_duration = sum(duration for _, _, _, duration in results)
        avg_duration = total_duration / len(results) if results else 0
        completed_tasks = sum(1 for _, _, result, _ in results if not result.startswith("Error:"))
        
        print("\n=== Batch Processing Statistics ===")
        print(f"Total tasks: {len(results)}")
        print(f"Successfully completed: {completed_tasks}")
        print(f"Failed: {len(results) - completed_tasks}")
        print(f"Total processing time: {total_duration:.2f} seconds")
        print(f"Average response time: {avg_duration:.2f} seconds per task")
        print(f"Effective throughput: {len(results) / total_duration:.2f} tasks per second")
        
        # Usage tracking with new API
        print("\nTotal usage across all batch operations is available through models.usage API")
    except Exception as e:
        logger.exception(f"Error in batch processing pattern: {e}")


def custom_model_pattern():
    """Demonstrates adding custom models to the registry."""
    try:
        # With the new API, we register models directly with the registry
        from ember.api.models import register_model
        
        # Register a custom model with realistic values
        custom_model = ModelInfo(
            id="custom:my-advanced-llm",
            name="MyOrg Advanced LLM",
            cost=ModelCost(
                input_cost_per_thousand=0.0015,  # $0.0015 per 1K input tokens
                output_cost_per_thousand=0.002   # $0.002 per 1K output tokens
            ),
            rate_limit=RateLimit(
                tokens_per_minute=100000,      # 100K tokens per minute
                requests_per_minute=3000       # 3K requests per minute
            ),
            context_window=128000,            # 128K context window
            provider={
                "name": "MyOrg AI",
                "default_api_key": "${MYORG_API_KEY}",
                "api_base": "https://api.myorg-ai.example.com/v1"
            }
        )
        
        # Register the model
        register_model(custom_model)
        
        # List all models
        model_ids = models.registry.list_models()
        
        print("\n=== Custom model registration ===")
        print(f"Registered models: {model_ids}")
        
        # Check model exists and get info
        exists = models.registry.model_exists("custom:my-advanced-llm")
        if exists:
            info = models.registry.get_model_info("custom:my-advanced-llm")
            print("\nCustom model details:")
            print(f"ID: {info.id}")
            print(f"Name: {info.name}")
            print(f"Provider: {info.provider['name']}")
            print(f"API Base URL: {info.provider.get('api_base', 'N/A')}")
            print(f"Context window: {info.context_window} tokens")
            print(f"Input cost: ${info.cost.input_cost_per_thousand:.4f} per 1K tokens")
            print(f"Output cost: ${info.cost.output_cost_per_thousand:.4f} per 1K tokens")
            print(f"Rate limits: {info.rate_limit.tokens_per_minute} tokens/min, {info.rate_limit.requests_per_minute} req/min")
        else:
            print("Custom model registration failed!")
    except Exception as e:
        logger.exception(f"Error in custom model pattern: {e}")


def main():
    """Run all example patterns."""
    print("Running Model Registry examples with the new API...\n")
    print("Make sure you have set up your API keys in environment variables:")
    print("  - OPENAI_API_KEY")
    print("  - ANTHROPIC_API_KEY")
    print("  - GOOGLE_API_KEY (if using Gemini models)")
    
    # Run each pattern
    one_line_pattern()
    standard_pattern()
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