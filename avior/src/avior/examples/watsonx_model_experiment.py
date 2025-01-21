"""
WatsonX Model Experiment Example

Usage:
    Quick start setup:
        pip install -e .
        cat <<"EOF" >> ./config.ini.watson
        [models]
        watsonx_api_key = MY_WATSONX_API_KEY
        watsonx_url = MY_WATSONX_MODEL_ENDPOINT
        watsonx_project_id = MY_WATSONX_PROJECT_ID
        watsonx_models = <Comma separated list of model found: 
                            https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx>
        EOF
        
        export AVIOR_CONFIG=MY_WATSONX_CONFIG_FILE
        export AVIOR_CUSTOM_MODEL=MY_WATSONX_EXPERIMENT_MODEL
 
    Run experiment:
        python examples/watsonx_model_experiment.py

Example:

pip install -e .
touch ./config.ini.watson && rm ./config.ini.watson

cat <<"EOF" >> ./config.ini.watsonx
[models]
watsonx_api_key = h204ejlkjsab3523yd44gs2YxR9V2
watsonx_url = https://us-south.ml.cloud.ibm.com
watsonx_project_id = cad89k3af-7752-43adc-a734-s124s4537o84
watsonx_models = ibm/granite-13b-instruct-v2, ibm/granite-13b-chat-v2
EOF

export AVIOR_CONFIG=config.ini.watsonx
export AVIOR_CUSTOM_MODEL=ibm/granite-13b-chat-v2

python examples/watsonx_model_experiment.py 
"""

import logging
import os
import sys

from avior.registry.model_registry import model_registry
from avior.core.configs.config import get_config

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


def define_non_graph(custom_model: str):
    """Defining and parsing NoN graph."""
    logger.info("Defining and parsing NoN graph")
    graph_def = [[f"3:E:{custom_model}"], f"1:MC:{custom_model}"]
    non_graph = NoNGraph().parse_from_list(graph_def)
    return non_graph


def valid_config_file() -> bool:
    # Check for 'watsonx_api_key'
    if get_config("models", "watsonx_api_key", fallback=None) is None:
        print(
            "Configuration variable 'watsonx_api_key' is not set in '[models]' stanza of config file."
        )
        return False

    # Check for 'watsonx_project_id'
    if get_config("models", "watsonx_project_id", fallback=None) is None:
        print(
            "Configuration variable 'watsonx_project_id' is not set in '[models]' stanza of config file."
        )
        return False

    # Check for 'watsonx_models'
    if get_config("models", "watsonx_models", fallback=None) is None:
        print(
            "Configuration variable 'watsonx_models' is not set in '[models]' stanza of config file."
        )
        return False

    return True


def main():
    """Main function to run the MCQ experiment."""
    logger.setLevel(logging.INFO)
    custom_model = os.getenv("AVIOR_CUSTOM_MODEL")

    if not custom_model:
        print("Env variable 'AVIOR_CUSTOM_MODEL' is not set.")
        sys.exit(1)

    if not valid_config_file():
        sys.exit(1)

    registry = model_registry.get_model_registry()
    model_list = registry.list_models()
    if custom_model not in model_list:
        print(f"Registered Models: {model_list}.")
        print(f"WatsonX {custom_model} is not registered.")
        sys.exit(1)

    non_graph = define_non_graph(custom_model)

    operator = OperatorContext(
        query="What is the capital of India?",
    )
    print(f"Sending {operator} to NoN: {non_graph}")
    response = non_graph.forward(operator)

    print(f"Final Answer = {response.final_answer}")


if __name__ == "__main__":
    main()
