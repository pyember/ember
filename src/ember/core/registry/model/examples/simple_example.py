"""Simple API usage example without discovery.

This example demonstrates basic usage of the model API without discovery.
"""

import logging

from ember.api import ModelBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Demonstrate the basic model API."""
    try:
        # Use the model builder to create a configured model
        model = ModelBuilder().temperature(0.7).max_tokens(50).build("openai:gpt-4o")

        # Generate a response
        response = model.generate(prompt="What is the capital of France?")
        print(f"Capital of France: {response.data}")

    except Exception as error:
        logger.exception("Error during API usage: %s", error)


if __name__ == "__main__":
    main()
