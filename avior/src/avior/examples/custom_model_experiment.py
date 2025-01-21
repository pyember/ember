"""
Custom Model Experiment Example

Usage:
    export AVIOR_API_KEY=avior_api_key
    export AVIOR_BASE_URL=http://avior_base_url
    export AVIOR_CUSTOM_MODEL=custom_model
    python custom_model_experiment.py

Example:
    python custom_model_experiment.py 
"""

import logging
import os
import sys

# Adding the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

from avior.registry.model_registry import model_registry
from avior.core.non_graph import NoNGraph
from avior.registry.operators.operator_base import OperatorContext
from avior.registry.model_registry.model_registry import (
    ModelInfo,
    ProviderInfo,
    ModelCost,
    RateLimit,
    OpenAIModel,
)

req_env_vars = ["AVIOR_CUSTOM_MODEL", "AVIOR_API_KEY", "AVIOR_BASE_URL"]


def define_non_graph(custom_model: str):
    """Defining and parsing NoN graph."""
    logger.info("Defining and parsing NoN graph")
    graph_def = [[f"3:E:{custom_model}"], f"1:MC:{custom_model}"]
    non_graph = NoNGraph().parse_from_list(graph_def)
    return non_graph


def main():
    """Main function to run the MCQ experiment."""
    logger.setLevel(logging.INFO)
    custom_model = os.getenv("AVIOR_CUSTOM_MODEL")
    base_url = os.getenv("AVIOR_BASE_URL")
    api_key = os.getenv("AVIOR_API_KEY")

    if not custom_model:
        print("Env variable 'AVIOR_CUSTOM_MODEL' is not set.")
        sys.exit(1)
    if not base_url:
        print("Env variable 'AVIOR_BASE_URL' is not set.")
        sys.exit(1)
    if not api_key:
        print("Env variable 'AVIOR_API_KEY' is not set.")
        sys.exit(1)

    custom_model_info = ModelInfo(
        model_id=custom_model,
        model_name=custom_model,
        cost=ModelCost(input_cost_per_thousand=0.0, output_cost_per_thousand=0.0),
        rate_limit=RateLimit(tokens_per_minute=0, requests_per_minute=0),
        provider=ProviderInfo(name="foundry", default_api_key="", base_url=base_url),
        api_key=api_key,
    )

    # Register the custom model info with OpenAI API endpoint
    model_registry.get_model_registry().register_model(
        custom_model_info, OpenAIModel(custom_model_info)
    )
    print(f"{model_registry.get_model_registry().list_models()}")
    non_graph = define_non_graph(custom_model)

    operator = OperatorContext(
        query="What is the capital of India?",
    )
    print(f"Sending {operator} to NoN: {non_graph}")
    response = non_graph.forward(operator)

    print(f"Final Answer = {response.final_answer}")


if __name__ == "__main__":
    main()
