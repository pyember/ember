import logging
from typing import List

from ember import initialize_ember
from ember.core.registry.model import load_model, ChatResponse
from ember.core.registry.model.base.services.model_service import ModelService

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    """Demonstrate Ember models usage patterns with robust error handling.

    This function initializes the Ember model registry with explicit configuration,
    creates a ModelService instance, and tests multiple model invocations using both
    service-based and direct approaches.

    Raises:
        Exception: Propagates any unhandled initialization errors.
    """
    try:
        # Initialize the registry of models from the merged YAML configuration.
        registry = initialize_ember(auto_register=True, auto_discover=True)

        # Create a ModelService instance.
        service = ModelService(registry=registry)

        # Define a list of model IDs to test — note that one of them is intentionally invalid.
        model_ids: List[str] = [
            "openai:o1",
            "openai:gpt-4o",
            "openai:gpt-4o-mini",
            "anthropic:claude-3.5-sonnet",
            "invalid:model",  # Expected to trigger an error.
            "deepmind:gemini-1.5-pro",
        ]

        for model_id in model_ids:
            try:
                logger.info("➡️ Testing model: %s", model_id)

                # Two usage styles are demonstrated below:
                # 1. Service-based invocation: Recommended for automatic usage tracking.
                service_response: ChatResponse = service.invoke_model(
                    model_id=model_id,
                    prompt="Explain quantum computing in 50 words",
                )
                print(f"🛎️ Service response from {model_id}:\n{service_response.data}\n")

                # 2. Direct model instance usage: Useful for more granular or PyTorch-like workflows.
                model = load_model(model_id=model_id, registry=registry)
                direct_response: ChatResponse = model(
                    prompt="What's the capital of France?"
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
