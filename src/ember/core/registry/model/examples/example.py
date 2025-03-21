"""Model registry usage examples.

This module demonstrates different ways to use the model registry and invoke models.
"""

import logging
import os
from typing import List

from ember.core.registry.model.base.schemas.chat_schemas import ChatResponse
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.initialization import initialize_registry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

# Get timeout from environment or use default
DEFAULT_TIMEOUT: int = int(os.environ.get("TIMEOUT", "30"))


def main() -> None:
    """Demonstrate model registry usage patterns.

    This function initializes the model registry, creates a ModelService instance,
    and tests different model invocation approaches.

    Raises:
        Exception: Propagates initialization errors.
    """
    try:
        # Initialize registry with discovery
        registry = initialize_registry(auto_discover=True, force_discovery=True)

        # Create a ModelService instance
        service = ModelService(registry=registry)

        # Define models to test - using string IDs for consistency
        test_models: List[str] = [
            "openai:gpt-4-turbo",
            "openai:gpt-4o",
            "openai:gpt-4o-mini",
            "anthropic:claude-3-5-sonnet",
            "invalid:model",  # Expected to trigger an error
            "deepmind:gemini-1.5-pro",
        ]

        for model_identifier in test_models:
            try:
                model_id = model_identifier
                logger.info("➡️ Testing model: %s", model_id)

                # Style 1: Service-based invocation with automatic usage tracking
                service_response: ChatResponse = service.invoke_model(
                    model_id=model_identifier,
                    prompt="Explain quantum computing in 50 words",
                    timeout=DEFAULT_TIMEOUT,
                )
                print(
                    f"🛎️ Service response from {model_id}:\n{service_response.data}\n"
                )

                # Style 2: Direct model instance for PyTorch-like workflows
                model = service.get_model(model_identifier)
                direct_response: ChatResponse = model(
                    prompt="What's the capital of France?",
                    timeout=DEFAULT_TIMEOUT,
                )
                print(f"🎯 Direct response from {model_id}:\n{direct_response.data}\n")

            except Exception as error:
                logger.error("❌ Error with model %s: %s", model_id, str(error))
                continue

    except Exception as error:
        logger.critical("🔥 Critical initialization error: %s", str(error))
        raise


if __name__ == "__main__":
    main()
