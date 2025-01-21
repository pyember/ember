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

import argparse
import logging
import os
import sys
from avior.core.non_graph import NoNGraph
from avior.registry.operators.operator_base import OperatorContext
from avior.registry.operators.operator_registry import OperatorFactory
from avior.registry.prompt_signature.prompt_registry import (
    PromptGenerator,
    PromptNames,
    PromptSpec,
)
from avior.registry.model_registry.model_registry import (
    ModelInfo,
    ProviderInfo,
    ModelCost,
    RateLimit,
    OpenAIModel,
)

# Adding the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

CARAVAN_PROMPT = "CARAVAN_PROMPT"
PRECISE_PROMPT = "PRECISE_PROMPT"

SIMPLE_NON = "simple"
CARAVAN_NON = "caravan"

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

sample_flow_stream = " (1) 0.001104000024497509,120.0,146.0,178.0,31.0,29.0,528985.5,644927.5,2.0,2.0,73.0,89.0,0.010999999940395355,0.009999999776482582,0.0,0.0,0.0,2.0,1.0,1.0 (2) 0.0009689999860711396,119.0,132.0,164.0,31.0,29.0,544891.625,676986.625,2.0,2.0,66.0,82.0,0.004000000189989805,0.010999999940395355,0.0,0.0,0.0,8.0,4.0,2.0 (3) 3.000000106112566e-06,119.0,114.0,0.0,254.0,0.0,152000000.0,0.0,2.0,0.0,57.0,0.0,0.003000000026077032,0.0,0.0,0.0,0.0,8.0,8.0,8.0 (4) 9.000000318337698e-06,119.0,264.0,0.0,60.0,0.0,117333328.0,0.0,2.0,0.0,132.0,0.0,0.008999999612569809,0.0,0.0,0.0,0.0,12.0,12.0,25.0 (5) 4.999999873689376e-06,119.0,114.0,0.0,254.0,0.0,91200000.0,0.0,2.0,0.0,57.0,0.0,0.004999999888241291,0.0,0.0,0.0,0.0,22.0,22.0,31.0 (6) 1.1568700075149536,113.0,1684.0,10168.0,31.0,29.0,10815.3896484375,66413.6875,14.0,18.0,120.0,565.0,88.96299743652344,68.01847076416016,0.0007060000207275152,0.0005520000122487545,0.0001539999939268455,4.0,4.0,1.0 (7) 0.0017600000137463212,119.0,528.0,304.0,31.0,29.0,1800000.0,1036363.625,4.0,4.0,132.0,76.0,0.45466700196266174,0.19200000166893005,0.0,0.0,0.0,9.0,3.0,6.0 (8) 0.0069240001030266285,113.0,3680.0,2456.0,31.0,29.0,4016175.5,2680531.5,18.0,18.0,204.0,136.0,0.3875879943370819,0.37882399559020996,0.0006150000263005495,0.0004799999878741801,0.00013499999477062374,4.0,5.0,3.0 (9) 0.005369000136852264,120.0,568.0,320.0,31.0,29.0,634755.0625,357608.46875,4.0,4.0,142.0,80.0,1.255666971206665,1.277999997138977,0.0,0.0,0.0,4.0,4.0,3.0 (10) 0.5125219821929932,114.0,8928.0,320.0,31.0,29.0,129414.9375,4167.6259765625,14.0,6.0,638.0,53.0,39.424766540527344,102.36280059814453,0.0007179999956861138,0.0005740000051446259,0.00014400000509340316,6.0,6.0,5.0 "


def define_non_graph(custom_model: str):
    """Defining and parsing NoN graph."""
    logger.info("Defining and parsing NoN graph")
    graph_def = [[f"3:E:{custom_model}"], f"1:VBJ:{custom_model}"]
    non_graph = NoNGraph().parse_from_list(graph_def)
    return non_graph


def parse_arguments() -> argparse.Namespace:
    """Parses and returns the command-line arguments.

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--non", type=str, default="simple", help="Simple/Caravan NoN")
    return parser.parse_args()


class CaravanNoN(NoNGraph):
    def __init__(self, model_name):
        super().__init__()
        caravan_prompts: dict[str, PromptSpec] = {}
        caravan_prompts[CARAVAN_PROMPT] = PromptSpec(
            prompt_name=CARAVAN_PROMPT,
            prompt_template=(
                "You are an expert in network security. The user is now labeling a network intrusion detection dataset, and he wants to assign a binary label (benign or malicious) to each traffic flow in the dataset based on each flow's input features. He'll give you a few labeled flows for reference, and you will help him label another few unlabeled flows. Feel free to use your own expertise and the information the user gives you. These are the features of the input flows (provided by UNSW-NB15 dataset) and meanings of the features: dur(Record total duration),proto(Transaction protocol, which will be categorized),sbytes(Source to destination transaction bytes),dbytes(Destination to source transaction bytes),sttl(Source to destination time to live value),dttl(Destination to source time to live value),sload(Source bits per second),dload(Destination bits per second),spkts(Source to destination packet count),dpkts(Destination to source packet count),smean(Mean of the packet size transmitted by the src),dmean(Mean of the packet size transmitted by the dst),sinpkt(Source interpacket arrival time (mSec)),dinpkt(Destination interpacket arrival time (mSec)),tcprtt(TCP connection setup round-trip time),synack(TCP connection setup time, the time between the SYN and the SYN_ACK packets),ackdat(TCP connection setup time, the time between the SYN_ACK and the ACK packets),ct_src_ltm(No. of connections of the same source address in 100 connections according to the last time),ct_dst_ltm(No. of connections of the same destination address in 100 connections according to the last time),ct_dst_src_ltm(No of connections of the same source and the destination address in 100 connections according to the last time)."
                "To begin with, here are some labeled flows for your reference later. The last field is the binary label (0 for benign and 1 for malicious): (1) 0.000976,119.0,146.0,178.0,31.0,29.0,598360.6875,729508.1875,2.0,2.0,73.0,89.0,0.01,0.007,0.0,0.0,0.0,1.0,1.0,1.0,0.0 (2) 0.751156,113.0,608.0,646.0,254.0,252.0,5836.337891,6198.446289,10.0,10.0,61.0,65.0,83.461778,68.822555,0.138493,0.092118,0.046375,1.0,2.0,2.0,1.0 (3) 1.108163,113.0,608.0,646.0,254.0,252.0,3956.09668,4201.547852,10.0,10.0,61.0,65.0,123.129222,128.379625,0.121057,0.0805,0.040557,1.0,2.0,1.0,1.0 (4) 0.029765,114.0,2766.0,28906.0,31.0,29.0,726759.625,7600336.0,44.0,46.0,63.0,628.0,0.684674,0.649933,0.000624,0.000514,0.00011,4.0,2.0,1.0,0.0 (5) 3e-06,97.0,78.0,0.0,254.0,0.0,104000000.0,0.0,2.0,0.0,39.0,0.0,0.003,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0 (6) 0.00443,113.0,1036.0,2262.0,31.0,29.0,1715575.5,3745372.25,12.0,12.0,86.0,189.0,0.368909,0.341182,0.000824,0.000667,0.000157,5.0,3.0,2.0,0.0 (7) 0.064837,119.0,2304.0,2304.0,62.0,252.0,213211.5938,213211.5938,4.0,4.0,576.0,576.0,15.984,14.002667,0.0,0.0,0.0,4.0,4.0,4.0,1.0 (8) 0.160126,113.0,2158.0,1918.0,31.0,29.0,103368.5938,91078.27344,24.0,20.0,90.0,96.0,6.962,8.392053,0.000694,0.000549,0.000145,3.0,2.0,1.0,0.0 (9) 0.001024,119.0,130.0,162.0,31.0,29.0,507812.5,632812.5,2.0,2.0,65.0,81.0,0.008,0.006,0.0,0.0,0.0,1.0,7.0,2.0,0.0 (10) 6e-06,123.0,200.0,0.0,254.0,0.0,133333328.0,0.0,2.0,0.0,100.0,0.0,0.006,0.0,0.0,0.0,0.0,8.0,4.0,4.0,1.0 (11) 8e-06,78.0,200.0,0.0,254.0,0.0,100000000.0,0.0,2.0,0.0,100.0,0.0,0.008,0.0,0.0,0.0,0.0,3.0,2.0,5.0,1.0 (12) 11.049956,114.0,13038.0,531716.0,31.0,29.0,9398.045898,384052.9063,226.0,427.0,58.0,1245.0,49.109444,25.937743,0.000601,0.000473,0.000128,5.0,4.0,2.0,0.0 (13) 7e-06,119.0,264.0,0.0,60.0,0.0,150857136.0,0.0,2.0,0.0,132.0,0.0,0.007,0.0,0.0,0.0,0.0,27.0,27.0,27.0,0.0 (14) 5.447018,113.0,140142.0,698.0,254.0,252.0,203988.3125,961.994202,112.0,16.0,1251.0,44.0,48.247566,359.71,0.112129,0.051363,0.060766,2.0,1.0,1.0,1.0 (15) 0.00108,120.0,146.0,178.0,31.0,29.0,540740.75,659259.3125,2.0,2.0,73.0,89.0,0.011,0.011,0.0,0.0,0.0,5.0,1.0,1.0,0.0 (16) 0.032656,113.0,3406.0,37226.0,31.0,29.0,819696.25,8956884.0,56.0,56.0,61.0,665.0,0.598574,0.584745,0.000611,0.000492,0.000119,3.0,1.0,2.0,0.0 (17) 7e-06,98.0,200.0,0.0,254.0,0.0,114285712.0,0.0,2.0,0.0,100.0,0.0,0.007,0.0,0.0,0.0,0.0,4.0,3.0,7.0,1.0 (18) 3.155537,113.0,1252.0,1658.0,62.0,252.0,3042.271484,4013.263184,24.0,22.0,52.0,75.0,137.159436,145.049333,0.177542,0.109492,0.06805,2.0,1.0,28.0,1.0 (19) 1.348011,113.0,364.0,2604.0,62.0,252.0,1893.159546,13525.11231,8.0,8.0,46.0,326.0,192.573,178.248422,0.157394,0.09975,0.057644,3.0,3.0,4.0,1.0 (20) 0.007337,113.0,1920.0,4312.0,31.0,29.0,1998637.0,4506474.0,22.0,24.0,87.0,180.0,0.3485,0.295043,0.00066,0.000544,0.000116,1.0,3.0,1.0,0.0. Next, I will give you some unlabeled flows for labeling."
                "Please give me a label (0 for benign and 1 for malicious) for each of these unlabeled flows. No explanation or analysis needed, label only; One flow on each line. Format for each line: (flow number) label. "
                "{question}"
            ),
            required_inputs=["question"],
        )
        self.add_node(
            "node1",
            OperatorFactory.create(
                "E",
                [{"model_name": model_name, "count": 3}],
                prompt_name=CARAVAN_PROMPT,
                custom_prompts=caravan_prompts,
            ),
        )
        self.add_node(
            "node2",
            OperatorFactory.create("MC", [{"model_name": model_name, "count": 1}]),
            inputs=["node1"],
        )


class SimpleNoN(NoNGraph):
    def __init__(self, model_name):
        super().__init__()
        precise_prompts: dict[str, PromptSpec] = {}
        precise_prompts[PRECISE_PROMPT] = PromptSpec(
            prompt_name=PRECISE_PROMPT,
            prompt_template=(
                "Provide a precise answer to the following question, without any verbosity.\n"
                "QUESTION:{question}"
            ),
            required_inputs=["question"],
        )
        self.add_node(
            "node1",
            OperatorFactory.create(
                "E",
                [{"model_name": model_name, "count": 1}],
                prompt_name=PRECISE_PROMPT,
                custom_prompts=precise_prompts,
            ),
        )
        self.add_node(
            "node2",
            OperatorFactory.create("MC", [{"model_name": model_name, "count": 1}]),
            inputs=["node1"],
        )


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
    args = parse_arguments()
    if args.non == SIMPLE_NON:
        simple_non = SimpleNoN(custom_model)
        operator = OperatorContext(
            query="What is the capital of India?",
        )
        print(f"Sending {operator} to NoN: {simple_non}")
        response = simple_non.forward(operator)

        print(f"Final Answer = {response}")
    elif args.non == CARAVAN_NON:
        caravan_non = CaravanNoN(custom_model)
        caravan_operator = OperatorContext(query=sample_flow_stream)
        print(f"Sending {caravan_operator} to NoN: {caravan_non}")
        response = caravan_non.forward(caravan_operator)

        print(f"Final Answer = {response.final_answer}")
    else:
        print("Invalid NoNType")


if __name__ == "__main__":
    main()
