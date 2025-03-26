"""Mistral-Large-Instruct-2407 usage example.

This module demonstrates how to use the Mistral-Large-Instruct-2407 model
through the Ember framework, showcasing various capabilities including
function calling and structured output.
"""

import logging
import os
from typing import Any, Dict, Optional, Union, cast

from ember.core.registry.model.base.schemas.chat_schemas import ChatResponse
from ember.core.registry.model.base.schemas.model_info import ModelInfo, ProviderInfo
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.initialization import initialize_registry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model_service() -> ModelService:
    """Get the model service with Mistral-Large-Instruct-2407 registered.

    Returns:
        A ModelService instance with Mistral-Large-Instruct-2407 registered
    """
    # Initialize the registry
    registry = initialize_registry(auto_discover=True)
    service = ModelService(registry=registry)
    
    # Register Mistral-Large-Instruct-2407 if not already registered
    if not service.get_model("huggingface:mistralai/Mistral-Large-Instruct-2407"):
        # Get API key from environment variable or use a default for testing
        api_key = os.environ.get("HUGGINGFACE_API_KEY", "your_api_key_here")
        
        # Create model info
        model_info = ModelInfo(
            id="huggingface:mistralai/Mistral-Large-Instruct-2407",
            name="mistralai/Mistral-Large-Instruct-2407",
            provider=ProviderInfo(name="HuggingFace", default_api_key=api_key),
        )
        
        # Register the model
        service.register_model(model_info)
    
    return service


def basic_generation_example(service: ModelService) -> None:
    """Demonstrate basic text generation with Mistral-Large-Instruct-2407.
    
    Args:
        service: The model service with Mistral-Large registered
    """
    logger.info("Running basic generation example...")
    
    try:
        # Example 1: Using the service to invoke the model
        response = service.invoke_model(
            model_id="huggingface:mistralai/Mistral-Large-Instruct-2407",
            prompt="Explain quantum computing in simple terms.",
            temperature=0.7,
            max_tokens=256
        )
        
        print("\n=== Basic Generation ===")
        print(f"Response: {response.data}")
        print(f"Token usage: {response.usage.total_tokens} tokens")
        
    except Exception as error:
        logger.exception("Error during basic generation: %s", error)


def function_calling_example(service: ModelService) -> None:
    """Demonstrate function calling with Mistral-Large-Instruct-2407.
    
    Args:
        service: The model service with Mistral-Large registered
    """
    logger.info("Running function calling example...")
    
    try:
        # Get the model directly
        mistral_model = service.get_model("huggingface:mistralai/Mistral-Large-Instruct-2407")
        
        # Define a weather function
        weather_function = {
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use.",
                        },
                    },
                    "required": ["location", "format"],
                }
            }
        }
        
        # Call with function calling capabilities
        response = mistral_model(
            "What's the weather like today in Paris and New York?",
            temperature=0.2,  # Lower temperature for more deterministic outputs
            provider_params={"tools": [weather_function]}
        )
        
        print("\n=== Function Calling ===")
        print(f"Response: {response.data}")
        print(f"Raw output: {response.raw_output}")
        
    except Exception as error:
        logger.exception("Error during function calling: %s", error)


def structured_output_example(service: ModelService) -> None:
    """Demonstrate structured output with Mistral-Large-Instruct-2407.
    
    Args:
        service: The model service with Mistral-Large registered
    """
    logger.info("Running structured output example...")
    
    try:
        # Get the model directly
        mistral_model = service.get_model("huggingface:mistralai/Mistral-Large-Instruct-2407")
        
        # Define a JSON schema for structured output
        json_schema = {
            "type": "object",
            "properties": {
                "cities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "country": {"type": "string"},
                            "population": {"type": "integer"},
                            "famous_landmarks": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["name", "country", "population", "famous_landmarks"]
                    }
                }
            },
            "required": ["cities"]
        }
        
        # Call with structured output
        response = mistral_model(
            "List the 3 most populous cities in Europe with their famous landmarks.",
            temperature=0.3,
            provider_params={"grammar": {"type": "json", "value": json_schema}}
        )
        
        print("\n=== Structured Output ===")
        print(f"Response: {response.data}")
        
        # You could parse this as JSON
        # import json
        # structured_data = json.loads(response.data)
        # print(f"First city: {structured_data['cities'][0]['name']}")
        
    except Exception as error:
        logger.exception("Error during structured output: %s", error)


def main() -> None:
    """Run all Mistral-Large-Instruct-2407 examples."""
    print("Mistral-Large-Instruct-2407 Usage Examples")
    print("=========================================")
    
    try:
        # Get the model service with Mistral registered
        service = get_model_service()
        
        # Run examples
        basic_generation_example(service)
        function_calling_example(service)
        structured_output_example(service)
        
        print("\nAll examples completed successfully!")
        
    except Exception as error:
        logger.exception("Error during examples: %s", error)


if __name__ == "__main__":
    main() 