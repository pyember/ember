import logging

# Import the global config init, registry, and the ModelService
from src.avior.registry.model.config import (
    initialize_global_registry,
    GLOBAL_MODEL_REGISTRY,
    GLOBAL_USAGE_SERVICE,
)
from src.avior.registry.model.services.model_service import ModelService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Demonstrates usage of multiple models (OpenAI, Anthropic, DeepMind, etc.)
    by leveraging the global registry and ModelService.
    """
    # 1) Initialize the global registry (auto-registers models from config.yaml + included configs)
    initialize_global_registry()

    # 2) Create a ModelService that uses GLOBAL_MODEL_REGISTRY + GLOBAL_USAGE_SERVICE by default
    service = ModelService(registry=GLOBAL_MODEL_REGISTRY, usage_service=GLOBAL_USAGE_SERVICE)

    # 3) Define some model IDs we expect to be registered (based on your YAML configs)
    model_ids = [
        "openai:o1",
        "openai:gpt-4o",
        "openai:gpt-4o-mini",
        "anthropic:claude-3.5-sonnet",
        "google:gemini-1.5-pro",
    ]

    # 4) Iterate over each model ID to demonstrate usage
    for m_id in model_ids:
        try:
            logger.info(f"\n--- Invoking model: {m_id} ---")

            # (A) Service-based usage
            #     We pass the prompt and the model_id to the ModelService __call__ method
            prompt_text = f"Hello from {m_id}, can you introduce yourself?"
            response = service(prompt=prompt_text, model_id=m_id)
            print(f"Service-based response from '{m_id}': {response.data}")

            # (B) Direct usage from the registry
            #     If you need more control, you can fetch the model instance yourself.
            gpt4 = GLOBAL_MODEL_REGISTRY.get_model(m_id)
            if not gpt4:
                print(f"No model instance found in registry for '{m_id}'")
                continue

            direct_prompt = "What's your top recommendation for a productivity hack?"
            direct_response = gpt4(direct_prompt)
            print(f"Direct usage response from '{m_id}': {direct_response.data}")

        except Exception as e:
            logger.error(f"Error invoking '{m_id}': {e}")

    print("\nDone invoking all example models.")

if __name__ == "__main__":
    main()
