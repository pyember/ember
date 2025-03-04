import logging
from ember.core.registry.model.config.settings import initialize_ember as init

# Optional: import ModelEnum for safer, type-checked invocation if available.
try:
    from ember.core.registry.model.config.model_enum import ModelEnum
except ImportError:
    ModelEnum = None


def main() -> None:
    """
    This example demonstrates three ways of invoking a model:
      1. Using the single-step `init()` helper with a string identifier.
      2. Retrieving the model instance directly for "PyTorch-like" invocation.
      3. (Optional) Using an Enum to invoke the model for safer type-checking.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize the ModelService via the single-step helper.
    service = init(usage_tracking=True)

    # Example 1: Service-based invocation using a string model ID.
    try:
        response_str = service("openai:gpt-4o", "Hello from string ID!")
        print("Response using string ID:\n", response_str.data)
    except Exception as e:
        logger.exception("Error during string ID invocation: %s", e)

    # Example 2: Direct model invocation (PyTorch-like usage).
    try:
        gpt4o = service.get_model("openai:gpt-4o")
        response_direct = gpt4o("What is the capital of France?")
        print("Direct model call response:\n", response_direct.data)
    except Exception as e:
        logger.exception("Error during direct model invocation: %s", e)

    # Example 3: (Optional) Service-based invocation using an Enum.
    # This pattern provides safer type-checking if your application maintains enumerations.
    if ModelEnum is not None:
        try:
            response_enum = service(
                ModelEnum.OPENAI_GPT4, "Hello from Enum invocation!"
            )
            print("Response using Enum:\n", response_enum.data)
        except Exception as e:
            logger.exception("Error during enum-based invocation: %s", e)
    else:
        print("ModelEnum not available; skipping enum-based example.")


if __name__ == "__main__":
    main()
