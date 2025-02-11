import logging
from typing import Any, List, Union

from ember.core.registry.model.settings import (
    initialize_global_registry,
    GLOBAL_MODEL_REGISTRY,
)
from ember.core.registry.model.core.services.model_service import ModelService
from ember.core.registry.model.core.schemas.chat_schemas import ChatResponse
from ember.core.registry.model.provider_registry.base_provider import BaseProviderModel

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    """Demonstrate usage of multiple AI models via the global registry and ModelService.

    This function initializes the global registry, creates a ModelService instance,
    and iterates over a list of model IDs to execute and display responses using both
    service-based and direct model invocations.
    """
    # Step 1: Initialize the global registry with auto_discover=True for auto discovery of viable models
    initialize_global_registry(auto_discover=True)

    # Step 2: Create a ModelService, but skip usage_service if you don't need usage logs
    service: ModelService = ModelService(
        registry=GLOBAL_MODEL_REGISTRY,
        usage_service=None,  # <--- For simplicity, we can omit usage tracking and usage logging
    )

    # Step 3: Define a list of model IDs expected to be used for this demonstration.
    model_ids: List[str] = [
        "openai:o1",
        "openai:gpt-4o",
        "openai:gpt-4o-mini",
        "anthropic:claude-3.5-sonnet",
        "deepmind:gemini-1.5-pro",
    ]

    # Step 4: Iterate over each model ID to demonstrate invocation.
    for model_id in model_ids:
        try:
            logger.info("Invoking model: %s", model_id)

            # (A) Service-based usage: Invoke the service with named parameters.
            prompt_text: str = f"Hello from {model_id}, can you introduce yourself?"
            response: ChatResponse = service(prompt=prompt_text, model_id=model_id)
            print(f"Service-based response from '{model_id}': {response.data}")

            # (B) Direct usage from the registry: Retrieve the model instance for direct invocation.
            model_instance: Union[BaseProviderModel, None] = (
                GLOBAL_MODEL_REGISTRY.get_model(model_id)
            )
            if model_instance is None:
                print("No model instance found in registry for '%s'" % model_id)
                continue

            direct_prompt: str = (
                "What's your top recommendation for a productivity hack?"
            )
            direct_response: ChatResponse = model_instance(prompt=direct_prompt)
            print(
                "Direct usage response from '%s': %s" % (model_id, direct_response.data)
            )

        except Exception as exc:
            logger.error("Error invoking model '%s': %s", model_id, exc)

    print("\nDone invoking all example models.")


if __name__ == "__main__":
    main()
