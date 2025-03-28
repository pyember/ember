"""Model usage patterns demonstration.

This module demonstrates several ways to use models through the model registry.
"""

import logging
from typing import Any, Optional, cast

from ember.core.registry.model.base.schemas.chat_schemas import ChatResponse
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.initialization import initialize_registry

# This will be monkeypatched in tests
model_service_factory: Optional[Any] = None


def get_model_service(initialize_context: bool = True) -> ModelService:
    """Get the model service, either from the factory or by creating a new one.

    This function exists for testability, allowing tests to inject a mock service.

    Args:
        initialize_context: Whether to initialize the application context

    Returns:
        A ModelService instance
    """
    # For testing: use the monkeypatched factory if available
    if model_service_factory is not None:
        # We know it will return a ModelService, but mypy doesn't
        service = model_service_factory(initialize_context=initialize_context)
        return cast(ModelService, service)

    # Normal case: create a registry and service
    registry = initialize_registry(auto_discover=True)
    return ModelService(registry=registry)


def main() -> None:
    """Demonstrate various model invocation patterns.

    This example shows three ways of invoking a model:
    1. Using a string identifier with the service
    2. Getting a model instance for direct invocation
    3. Using ModelEnum for type-safe invocation
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Get the model service
    service = get_model_service(initialize_context=True)
    print(service.list_models())
    print(service.get_model_info("openai:gpt-4o"))

    # The ModelService provides access to the registry methods
    # No need to directly access the registry attribute

    # Example 1: String-based invocation with error handling
    try:
        response_str: ChatResponse = service.invoke_model(
            model_id="openai:gpt-4o", prompt="Hello from string ID!"
        )
        print(f"Response using string ID:\n{response_str.data}")
    except Exception as error:
        logger.exception("Error during string ID invocation: %s", error)

    # Example 2: Direct model invocation
    try:
        gpt4o = service.get_model("openai:gpt-4o")
        response_direct: ChatResponse = gpt4o(prompt="What is the capital of France?")
        print(f"Direct model call response:\n{response_direct.data}")
    except Exception as error:
        logger.exception("Error during direct model invocation: %s", error)

    # Example 3: Enum-based invocation for type safety
    try:
        response_enum: ChatResponse = service.invoke_model(
            model_id="openai:gpt-4o",  # Use string instead of enum for now
            prompt="Hello from Enum invocation!",
        )
        print(f"Response using Enum:\n{response_enum.data}")
    except Exception as error:
        logger.exception("Error during enum-based invocation: %s", error)


if __name__ == "__main__":
    main()
