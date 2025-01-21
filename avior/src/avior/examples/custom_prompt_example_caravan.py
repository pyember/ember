#!/usr/bin/env python3
"""
Refactored Custom Prompt Example with Old Context

Usage:
    export AVIOR_API_KEY=avior_api_key
    export AVIOR_BASE_URL=http://avior_base_url
    export AVIOR_CUSTOM_MODEL=custom_model
    python custom_prompt_example_caravan.py --non simple

Example:
    python custom_prompt_example_caravan.py --non caravan

Overview:
    1) 'simple': minimal single-sentence Q&A pipeline.
    2) 'caravan': more advanced prompt that references the UNSW-NB15 dataset
                  flows, providing labeled references and then labeling new flows.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any, Optional, List, Type, Union

from pydantic import BaseModel, model_validator

# ------------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------
# Constants & Env
# ------------------------------------------------------------------------------------
req_env_vars = ["AVIOR_CUSTOM_MODEL", "AVIOR_API_KEY", "AVIOR_BASE_URL"]
SIMPLE_NON = "simple"
CARAVAN_NON = "caravan"

# A larger sample flow stream from your prior code references:
sample_flow_stream = (
    " (1) 0.001104000024497509,120.0,146.0,178.0,31.0,29.0,528985.5,644927.5,2.0,2.0,73.0,89.0,"
    "0.010999999940395355,0.009999999776482582,0.0,0.0,0.0,2.0,1.0,1.0 "
    "(2) 0.0009689999860711396,119.0,132.0,164.0,31.0,29.0,544891.625,676986.625,2.0,2.0,66.0,"
    "82.0,0.004000000189989805,0.010999999940395355,0.0,0.0,0.0,8.0,4.0,2.0 "
    "(3) 3.000000106112566e-06,119.0,114.0,0.0,254.0,0.0,152000000.0,0.0,2.0,0.0,57.0,0.0,"
    "0.003000000026077032,0.0,0.0,0.0,0.0,8.0,8.0,8.0 "
    "(4) 9.000000318337698e-06,119.0,264.0,0.0,60.0,0.0,117333328.0,0.0,2.0,0.0,132.0,0.0,"
    "0.008999999612569809,0.0,0.0,0.0,0.0,12.0,12.0,25.0 "
    "(5) 4.999999873689376e-06,119.0,114.0,0.0,254.0,0.0,91200000.0,0.0,2.0,0.0,57.0,0.0,"
    "0.004999999888241291,0.0,0.0,0.0,0.0,22.0,22.0,31.0 "
    "(6) 1.1568700075149536,113.0,1684.0,10168.0,31.0,29.0,10815.3896484375,66413.6875,14.0,"
    "18.0,120.0,565.0,88.96299743652344,68.01847076416016,0.0007060000207275152,"
    "0.0005520000122487545,0.0001539999939268455,4.0,4.0,1.0 "
    "(7) 0.0017600000137463212,119.0,528.0,304.0,31.0,29.0,1800000.0,1036363.625,4.0,4.0,"
    "132.0,76.0,0.45466700196266174,0.19200000166893005,0.0,0.0,0.0,9.0,3.0,6.0 "
    "(8) 0.0069240001030266285,113.0,3680.0,2456.0,31.0,29.0,4016175.5,2680531.5,18.0,18.0,"
    "204.0,136.0,0.3875879943370819,0.37882399559020996,0.0006150000263005495,"
    "0.0004799999878741801,0.00013499999477062374,4.0,5.0,3.0 "
    "(9) 0.005369000136852264,120.0,568.0,320.0,31.0,29.0,634755.0625,357608.46875,4.0,4.0,"
    "142.0,80.0,1.255666971206665,1.277999997138977,0.0,0.0,0.0,4.0,4.0,3.0 "
    "(10) 0.5125219821929932,114.0,8928.0,320.0,31.0,29.0,129414.9375,4167.6259765625,"
    "14.0,6.0,638.0,53.0,39.424766540527344,102.36280059814453,0.0007179999956861138,"
    "0.0005740000051446259,0.00014400000509340316,6.0,6.0,5.0 "
)

# ------------------------------------------------------------------------------------
# Environment Validation
# ------------------------------------------------------------------------------------
def check_env() -> None:
    """Ensure all required environment variables are set."""
    missing = [e for e in req_env_vars if not os.getenv(e)]
    if missing:
        logger.error(f"Missing env vars: {missing}")
        sys.exit(1)

# ------------------------------------------------------------------------------------
# Model Registration
# ------------------------------------------------------------------------------------
# Example model registry references
from src.avior.registry.model.registry.model_registry import ModelRegistry
from src.avior.registry.model.schemas.model_info import ModelInfo
from src.avior.registry.model.schemas.provider_info import ProviderInfo
from src.avior.registry.model.schemas.cost import ModelCost, RateLimit
from avior.registry.model.provider_registry.openai.openai_provider import OpenAIModel

def register_custom_model() -> None:
    """
    Registers the user-specified custom model with a local ModelRegistry instance.
    This must be done before creating any LMModule referencing `AVIOR_CUSTOM_MODEL`.
    """
    custom_model = os.getenv("AVIOR_CUSTOM_MODEL", "")
    base_url = os.getenv("AVIOR_BASE_URL", "")
    api_key = os.getenv("AVIOR_API_KEY", "")

    registry = ModelRegistry()
    model_info = ModelInfo(
        model_id=custom_model,
        model_name=custom_model,
        cost=ModelCost(input_cost_per_thousand=0.0, output_cost_per_thousand=0.0),
        rate_limit=RateLimit(tokens_per_minute=0, requests_per_minute=0),
        provider=ProviderInfo(
            name="foundry",
            default_api_key=api_key, 
            base_url=base_url
        ),
        api_key=api_key,
    )
    registry.register_model(model_info)
    logger.info(
        f"Registered custom model '{custom_model}' with base_url='{base_url}'. "
        f"Models now in registry: {registry.list_models()}"
    )

# ------------------------------------------------------------------------------------
# Prompt Pieces (Old Context, Broken Down)
# ------------------------------------------------------------------------------------
CARAVAN_PROMPT_INTRO = (
    "You are an expert in network security. The user is now labeling a network intrusion "
    "detection dataset (UNSW-NB15). He wants to assign a binary label (0=benign, 1=malicious) "
    "to each traffic flow based on its features."
)

CARAVAN_PROMPT_FEATURES = (
    "Features include: dur (duration), proto (protocol), sbytes/dbytes (src->dst/dst->src bytes), "
    "sttl/dttl (time to live), sload/dload (bits/sec), spkts/dpkts (packet counts), smean/dmean "
    "(mean packet sizes), sinpkt/dinpkt (interpacket arrival times), tcprtt/synack/ackdat "
    "(TCP handshake times), ct_src_ltm/ct_dst_ltm/ct_dst_src_ltm (connection counts), etc."
)

CARAVAN_PROMPT_REFERENCES = (
    "He provides some labeled flows for reference (the last field is the binary label). "
    "Next, he'll provide unlabeled flows and wants you to give a label for each, with no explanation."
)

CARAVAN_PROMPT_INSTRUCTIONS = (
    "Please output a label (0 or 1) per line in the format: (flow number) label. "
    "No explanation or analysis needed, label only."
)

CARAVAN_PROMPT_FULL = (
    f"{CARAVAN_PROMPT_INTRO}\n{CARAVAN_PROMPT_FEATURES}\n{CARAVAN_PROMPT_REFERENCES}\n"
    f"{CARAVAN_PROMPT_INSTRUCTIONS}\n"
    "UNLABELED FLOWS:\n"
    "{question}\n"
)

# ------------------------------------------------------------------------------------
# Minimal 'Signature' & 'Inputs' for Our Caravan Prompt
# ------------------------------------------------------------------------------------
class CaravanLabelingInputs(BaseModel):
    """
    The request object containing the unlabeled flows in 'question'.
    """
    question: str

class CaravanLabelingSignature(BaseModel):
    """
    Minimal local signature container with prompt string, required fields, etc.
    """
    required_inputs: List[str] = ["question"]
    prompt_template: str = CARAVAN_PROMPT_FULL

    def render_prompt(self, inputs: Dict[str, Any]) -> str:
        """
        Replace placeholders in the prompt template with actual values.
        """
        # For production usage, you'd ensure all required fields are present.
        return self.prompt_template.format(**inputs)

# ------------------------------------------------------------------------------------
# A Simple 'Signature' & 'Inputs' for the "simple" pipeline
# ------------------------------------------------------------------------------------
class SimplePromptInputs(BaseModel):
    """
    The request for a simple question like "What is the capital of India?"
    """
    question: str

class SimplePromptSignature(BaseModel):
    required_inputs: List[str] = ["question"]
    prompt_template: str = (
        "Provide a concise single-sentence answer to the following question:\n"
        "QUESTION: {question}\n"
    )

    def render_prompt(self, inputs: Dict[str, Any]) -> str:
        return self.prompt_template.format(**inputs)
# ------------------------------------------------------------------------------------
# Operators (Single-step LM calls using these signatures)
# ------------------------------------------------------------------------------------
from src.avior.registry.operator.operator_base import Operator, OperatorContext, LMModule, LMModuleConfig
from src.avior.registry import non

class SimplePromptOperator(Operator[SimplePromptInputs, Dict[str, Any]]):
    """
    Single-step operator for 'simple' Q&A.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.signature = SimplePromptSignature()
        self.ensemble = non.Ensemble(num_units=1, model_name=model_name, temperature=0.2, max_tokens=64)

    def forward(self, inputs: SimplePromptInputs) -> Dict[str, Any]:
        # Validate input (in more advanced usage, you'd do thorough checks).
        prompt = self.signature.render_prompt(inputs.dict())
        # Single LM module for demonstration
        ensemble_inputs = non.EnsembleInputs(query=prompt)
        raw_answer = self.ensemble(ensemble_inputs).get("final_answer", "").strip()
        return {"final_answer": raw_answer}

class CaravanLabelingOperator(Operator[CaravanLabelingInputs, Dict[str, Any]]):
    """
    Operator that uses a big, multi-part 'Caravan' prompt to label flows 0 or 1.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.signature = CaravanLabelingSignature()
        self.ensemble = non.Ensemble(num_units=3, model_name=model_name, temperature=0.0, max_tokens=256)
        self.judge = non.Judge(model_name=model_name, temperature=0.0, max_tokens=256)

    def forward(self, inputs: CaravanLabelingInputs) -> Dict[str, Any]:
        prompt = self.signature.render_prompt(inputs.dict())
        ensemble_inputs = non.EnsembleInputs(query=prompt)
        ensemble_output = self.ensemble(ensemble_inputs)

        judge_inputs = non.JudgeInputs(
            query=prompt,
            responses=ensemble_output.get("responses", [])
        )
        judge_output = self.judge(judge_inputs)
        return {"final_answer": judge_output.get("final_answer", "").strip()}

# ------------------------------------------------------------------------------------
# Graph/Pipeline Constructors
# ------------------------------------------------------------------------------------
def create_simple_pipeline(model_name: str) -> Operator:
    """Returns a single-step operator for 'simple' usage."""
    return SimplePromptOperator(model_name)

def create_caravan_pipeline(model_name: str) -> Operator:
    """Returns a single-step operator for 'caravan' labeling usage."""
    return CaravanLabelingOperator(model_name)

# ------------------------------------------------------------------------------------
# Main + Arg Parsing
# ------------------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refactored custom prompt example (caravan).")
    parser.add_argument(
        "--non",
        type=str,
        default=SIMPLE_NON,
        help="Which pipeline to run: 'simple' or 'caravan'.",
    )
    return parser.parse_args()

def main():
    logger.info("Starting refactored custom prompt with old context ...")
    check_env()
    register_custom_model()

    args = parse_arguments()
    chosen_non = args.non.lower().strip()
    model_name = os.getenv("AVIOR_CUSTOM_MODEL", "")

    if chosen_non == SIMPLE_NON:
        operator = create_simple_pipeline(model_name)
        # Example question:
        question_data = "What is the capital of India?"
        context = OperatorContext(query=question_data)
        response = operator.forward(SimplePromptInputs(question=question_data))
        print(f"[SIMPLE] Final Answer:\n{response['final_answer']}\n")

    elif chosen_non == CARAVAN_NON:
        operator = create_caravan_pipeline(model_name)
        # We'll pass the flows into the 'question' field:
        flows = sample_flow_stream
        context = OperatorContext(query=flows)
        response = operator.forward(CaravanLabelingInputs(question=flows))
        print(f"[CARAVAN] Final Labeled Output:\n{response['final_answer']}\n")

    else:
        logger.error(f"Invalid --non={chosen_non}. Must be '{SIMPLE_NON}' or '{CARAVAN_NON}'.")
        sys.exit(1)


if __name__ == "__main__":
    main()