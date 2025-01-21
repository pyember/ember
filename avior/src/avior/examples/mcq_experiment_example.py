#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCQ Experiment (MMLU) with parallel per-question scoring, refactored to only use
the typed wrappers in `non.py` (and a new ValidateChoice wrapper) rather than
directly referencing registry operator classes.

Pipelines:
1) SingleModelBaseline:
   - non.Ensemble(num_units=1) => non.GetAnswer => ValidateChoice
2) MultiModelEnsemble:
   - non.Ensemble(num_units=3) => non.JudgeSynthesis => non.GetAnswer => ValidateChoice

Launch this experiment via:

    python -m src.avior.examples.mcq_experiment_example --config_name "abstract_algebra" --num_samples 5 --max_workers 4
"""

import argparse
import logging
import os
import sys
import concurrent.futures
from typing import Dict, Any, List, Tuple, Optional, Type

from pydantic import BaseModel, field_validator
from prettytable import PrettyTable

# Avior imports
# We intentionally avoid referencing operator_base or operator_registry directly!
# Instead, we only use `non` and our typed pipeline definitions.
from src.avior.registry import non
from src.avior.core.operator_graph import OperatorGraph
from src.avior.core.operator_graph_runner import OperatorGraphRunner
from src.avior.registry.model.registry.model_registry import GLOBAL_MODEL_REGISTRY
from src.avior.core.configs.config import CONFIG, initialize_system

# For dataset usage:
from src.avior.registry.dataset.base.models import DatasetEntry
from src.avior.registry.dataset.datasets.mmlu import MMLUConfig
from src.avior.registry.dataset.registry.service import DatasetService
from src.avior.registry.dataset.registry.metadata_registry import DatasetMetadataRegistry
from src.avior.registry.dataset.registry.loader_factory import DatasetLoaderFactory
from src.avior.registry.dataset.base.loaders import HuggingFaceDatasetLoader
from src.avior.registry.dataset.base.samplers import DatasetSampler
from src.avior.registry.dataset.base.validators import DatasetValidator
from src.avior.registry.dataset.registry.initialization import initialize_dataset_registry
from src.avior.core.scheduler import ExecutionPlan
from src.avior.registry.operator.operator_base import Operator, OperatorMetadata, OperatorType, LMModule
from src.avior.registry.prompt_signature.signatures import Signature
from src.avior.modules.lm_modules import LMModuleConfig


###############################################################################
# A) EnsureValidChoiceOperator
###############################################################################
class EnsureValidChoiceInputs(BaseModel):
    query: str
    partial_answer: str
    choices: Dict[str, str]

class EnsureValidChoiceSignature(Signature):
    required_inputs: List[str] = ["query", "partial_answer", "choices"]
    input_model: Type[BaseModel] = EnsureValidChoiceInputs
    prompt_template: str = (
        "We have a query:\n"
        "{query}\n\n"
        "We have a set of valid choices:\n"
        "{choices}\n\n"
        "We have a partial or malformed answer:\n"
        "{partial_answer}\n\n"
        "You must respond with EXACTLY one uppercase letter among the valid keys, "
        "or 'Invalid' if no mapping is possible.\n"
        "No additional explanation, just the letter or 'Invalid'."
    )

class EnsureValidChoiceOperator(Operator[EnsureValidChoiceInputs, Dict[str, Any]]):
    """
    Attempts to convert a partial or invalid answer into a valid choice key via LLM fallback.
    """

    metadata = OperatorMetadata(
        code="ENSURE_VALID_CHOICE",
        description="Refines or validates that final_answer is one of the valid choices, or tries to fix it via LM.",
        operator_type=OperatorType.RECURRENT,
        signature=EnsureValidChoiceSignature(),
    )

    def __init__(
        self,
        model_name: str = "openai:o1",
        temperature: float = 0.0,
        max_tokens: Optional[int] = 16,
        max_retries: int = 1,
        **kwargs
    ):
        super().__init__(name="EnsureValidChoiceOperator", signature=self.metadata.signature)
        lm_config = LMModuleConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.lm_modules = [LMModule(config=lm_config)]
        self._max_retries = max_retries

    def forward(self, inputs: EnsureValidChoiceInputs) -> Dict[str, Any]:
        if inputs.partial_answer in inputs.choices:
            return {"final_answer": inputs.partial_answer}

        logging.info(
            f"[EnsureValidChoiceOperator] Invalid choice '{inputs.partial_answer}'. "
            f"Valid keys are: {list(inputs.choices.keys())}. Attempting fallback..."
        )

        refined_answer = inputs.partial_answer
        for attempt_i in range(self._max_retries):
            prompt_context = {
                "query": inputs.query,
                "partial_answer": refined_answer,
                "choices": "\n".join(f"{k}: {v}" for k, v in inputs.choices.items()),
            }
            prompt_str = self.build_prompt(prompt_context)

            fallback_output = self.call_lm(prompt_str, self.lm_modules[0]).strip().upper()
            logging.info(f"Attempt {attempt_i+1}/{self._max_retries}, LM suggested: '{fallback_output}'")

            if fallback_output in inputs.choices:
                return {"final_answer": fallback_output}
            refined_answer = fallback_output

        raise ValueError(
            f"[EnsureValidChoiceOperator] Could not map answer '{inputs.partial_answer}' "
            f"to any valid choice {list(inputs.choices.keys())} after {self._max_retries} attempts."
        )

    def to_plan(self, inputs: EnsureValidChoiceInputs) -> Optional["ExecutionPlan"]:
        return None


###############################################################################
# 1) SingleModelBaseline pipeline
###############################################################################
class SingleModelBaselineInputs(BaseModel):
    query: str
    choices: Dict[str, str]

class SingleModelBaselineOutputs(BaseModel):
    final_answer: str

# 1) Define a signature referencing SingleModelBaselineInputs
class SingleModelBaselineSignature(Signature):
    required_inputs: List[str] = ["query", "choices"]
    input_model: Type[BaseModel] = SingleModelBaselineInputs

# 2) Attach that signature to the operator’s metadata
class SingleModelBaseline(non.Operator[SingleModelBaselineInputs, SingleModelBaselineOutputs]):
    metadata: OperatorMetadata = OperatorMetadata(
        code="SINGLE_MODEL_BASELINE",
        description="One-LM baseline with ValidateChoice or EnsureValidChoice.",
        operator_type=OperatorType.RECURRENT,
        signature=SingleModelBaselineSignature(),
    )

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        super().__init__(name="SingleModelBaseline", signature=self.metadata.signature)
        # 1) Ensemble
        self.ensemble = non.Ensemble(num_units=1, model_name=model_name, temperature=temperature)
        # 2) GetAnswer
        self.get_answer = non.GetAnswer(model_name=model_name, temperature=temperature)
        # 3) EnsureValidChoice (or ValidateChoice)
        self.ensure_valid_choice = EnsureValidChoiceOperator(
            model_name=model_name,
            temperature=0.0,
            max_tokens=16,
            max_retries=1,
        )

    def forward(self, inputs: SingleModelBaselineInputs) -> SingleModelBaselineOutputs:
        # 1) ensemble => {"responses": [...]}
        ens_out = self.ensemble({"query": inputs.query})
        responses = ens_out["responses"]

        # 2) get_answer => {"final_answer": ...}
        ga_out = self.get_answer({"query": inputs.query, "responses": responses})
        final_answer = ga_out["final_answer"]

        # 3) ensure_valid_choice => ensures final_answer is in `choices`
        evc_out = self.ensure_valid_choice({
            "query": inputs.query,
            "partial_answer": final_answer,
            "choices": inputs.choices
        })

        return SingleModelBaselineOutputs(final_answer=evc_out["final_answer"])


###############################################################################
# 2) MultiModelEnsemble pipeline
###############################################################################
class MultiModelEnsembleInputs(BaseModel):
    query: str
    choices: Dict[str, str]

class MultiModelEnsembleOutputs(BaseModel):
    final_answer: str

# Step A: Define a signature referencing MultiModelEnsembleInputs
class MultiModelEnsembleSignature(Signature):
    required_inputs: List[str] = ["query", "choices"]
    input_model: Type[BaseModel] = MultiModelEnsembleInputs

class MultiModelEnsemble(non.Operator[MultiModelEnsembleInputs, MultiModelEnsembleOutputs]):
    """
    Pipeline:
      1) non.Ensemble(num_units=3) => {'responses': [...]}
      2) non.JudgeSynthesis => {'final_answer': ...} (by synthesizing multiple answers)
      3) non.GetAnswer => (optional post-processing step)
      4) non.ValidateChoice => ensures final_answer is among input.choices
    """

    # Step B: Attach that signature to the operator’s metadata
    metadata: OperatorMetadata = OperatorMetadata(
        code="MULTI_MODEL_ENSEMBLE",
        description="Multi-model ensemble aggregator with judge step.",
        operator_type=OperatorType.RECURRENT,
        signature=MultiModelEnsembleSignature(),
    )

    def __init__(self, model_name: str = "claude-3.5-sonnet-latest", temperature: float = 0.7):
        # Step C: Pass the signature to super().__init__ so it can coerce inputs
        super().__init__(name="MultiModelEnsemble", signature=self.metadata.signature)

        # Step 1) Ensembling multiple LM calls
        self.ensemble = non.Ensemble(num_units=3, model_name=model_name, temperature=temperature)
        # Step 2) Judge-synthesis aggregator
        self.judge = non.JudgeSynthesis(model_name=model_name, temperature=temperature)
        # Step 3) (Optional) refine or parse answer again
        self.get_answer = non.GetAnswer(model_name=model_name, temperature=0.1)
        # Step 4) Validate final_answer is in choices
        self.ensure_valid_choice = EnsureValidChoiceOperator(
            model_name=model_name,
            temperature=0.1,
            max_tokens=16,
            max_retries=1
        )

    def forward(self, inputs: MultiModelEnsembleInputs) -> MultiModelEnsembleOutputs:
        # Step 1) ensemble => {"responses": [...]}
        ens_out = self.ensemble({"query": inputs.query})
        responses = ens_out["responses"]

        judge_out = self.judge({"query": inputs.query, "responses": responses})
        judge_answer = judge_out.final_answer

        # Step 3) run get_answer on the single judge response (optional)
        ga_out = self.get_answer({"query": inputs.query, "responses": [judge_answer]})
        final_answer = ga_out["final_answer"]

        # Step 4) validate final_answer
        evc_out = self.ensure_valid_choice({
            "query": inputs.query,
            "partial_answer": final_answer,
            "choices": inputs.choices
        })

        return MultiModelEnsembleOutputs(final_answer=evc_out["final_answer"])


###############################################################################
# 2b) VariedModelEnsemble pipeline (refactored to match MultiModelEnsemble)
###############################################################################
class VariedModelEnsembleInputs(BaseModel):
    query: str
    choices: Dict[str, str]

class VariedModelEnsembleOutputs(BaseModel):
    final_answer: str

class VariedModelEnsembleSignature(Signature):
    required_inputs: List[str] = ["query", "choices"]
    input_model: Type[BaseModel] = VariedModelEnsembleInputs

class VariedModelEnsemble(non.Operator[VariedModelEnsembleInputs, VariedModelEnsembleOutputs]):
    """
    Pipeline:
      1) non.VariedEnsemble => {'responses': [...]}
      2) non.JudgeSynthesis => {'final_answer': ...}
      3) non.GetAnswer => (optional post-processing step)
      4) non.ValidateChoice => ensures final_answer is among input.choices
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="VARIED_MODEL_ENSEMBLE",
        description="Multi-model pipeline aggregator with judge step, using VariedEnsemble for step #1.",
        operator_type=OperatorType.RECURRENT,
        signature=VariedModelEnsembleSignature(),
    )

    def __init__(self, model_configs: List[LMModuleConfig]):
        super().__init__(name="VariedModelEnsemble", signature=self.metadata.signature)

        # Use VariedEnsemble at step 1
        self.ensemble = non.VariedEnsemble(model_configs=model_configs)

        # Updated: read aggregator model details from 'model_id' instead of 'model_name'
        aggregator_model_name = 'openai:o1'
        aggregator_temp = 0.7

        self.judge = non.JudgeSynthesis(model_name=aggregator_model_name, temperature=aggregator_temp)
        self.get_answer = non.GetAnswer(model_name=aggregator_model_name, temperature=max(0.0, aggregator_temp - 0.6))
        self.ensure_valid_choice = EnsureValidChoiceOperator(
            model_name=aggregator_model_name, temperature=aggregator_temp, max_tokens=16, max_retries=1
        )

    def forward(self, inputs: VariedModelEnsembleInputs) -> VariedModelEnsembleOutputs:
        ens_out = self.ensemble({"query": inputs.query})
        responses = ens_out.responses

        judge_out = self.judge({"query": inputs.query, "responses": responses})
        judge_answer = judge_out.final_answer

        ga_out = self.get_answer({"query": inputs.query, "responses": [judge_answer]})
        final_answer = ga_out["final_answer"]

        evc_out = self.ensure_valid_choice({
            "query": inputs.query,
            "partial_answer": final_answer,
            "choices": inputs.choices
        })

        return VariedModelEnsembleOutputs(final_answer=evc_out["final_answer"])


###############################################################################
# 3) Build each pipeline as a one-node OperatorGraph
###############################################################################
def build_pipeline_graph(pipeline_op: non.Operator) -> OperatorGraph:
    """
    Wraps a single operator pipeline in an OperatorGraph with one node,
    letting us run it via OperatorGraphRunner.
    """
    graph = OperatorGraph()
    graph.add_node(pipeline_op, node_id=pipeline_op.name)
    return graph


###############################################################################
# 4) Main experiment code
###############################################################################
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCQ Experiment with Baseline vs. Multi-Model Ensemble.")
    parser.add_argument("--config_name", type=str, default="abstract_algebra",
                        help="MMLU sub-config (e.g. 'abstract_algebra').")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="How many MMLU samples to test on.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Number of threads for parallel scoring. Defaults to Python's choice.")
    return parser.parse_args()


def setup_openai_api_key() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        CONFIG.set("models", "openai_api_key", api_key)
        logging.info("OpenAI API key set from environment.")
    else:
        logging.warning("No OPENAI_API_KEY found; continuing without it.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_arguments()

    # Initialize system with the global model registry
    initialize_system(registry=GLOBAL_MODEL_REGISTRY)
    setup_openai_api_key()

    # (Optional) If needed, register your model in GLOBAL_MODEL_REGISTRY here
    # e.g. for "openai:gpt-4o"

    # Initialize MMLU dataset environment
    metadata_registry = DatasetMetadataRegistry()
    loader_factory = DatasetLoaderFactory()
    initialize_dataset_registry(metadata_registry, loader_factory)

    # Obtain MMLU dataset info and create a prepper
    mmlu_info = metadata_registry.get("mmlu")
    if not mmlu_info:
        logging.error("MMLU dataset not found. Ensure it is registered.")
        sys.exit(1)
    prepper_cls = loader_factory.get_prepper_class("mmlu")
    if not prepper_cls:
        logging.error("No MMLU prepper found in the registry.")
        sys.exit(1)

    mmlu_config = MMLUConfig(config_name=args.config_name, split="dev")
    prepper = prepper_cls(config=mmlu_config)

    dataset_service = DatasetService(
        loader=HuggingFaceDatasetLoader(),
        validator=DatasetValidator(),
        sampler=DatasetSampler(),
        transformers=[],
    )
    dataset_entries: List[DatasetEntry] = dataset_service.load_and_prepare(
        dataset_info=mmlu_info,
        prepper=prepper,
        config=mmlu_config,
        num_samples=args.num_samples
    )
    logging.info(f"Loaded {len(dataset_entries)} MMLU items (config={args.config_name}).")

    # Build the pipeline operators as typed wrappers only (no direct registry usage).
    baseline_op = SingleModelBaseline(model_name="gpt-4o", temperature=0.0)
    ensemble_op = MultiModelEnsemble(model_name="gpt-4o", temperature=0.7)
    varied_op = VariedModelEnsemble([
        LMModuleConfig(model_name="openai:gpt-4o", temperature=0.6),
        LMModuleConfig(model_name="anthropic:claude-3.5-sonnet-latest", temperature=0.8),
        LMModuleConfig(model_name="openai:o1", temperature=0.4),
        LMModuleConfig(model_name="google:gemini-1.5-pro", temperature=0.5),
    ])

    baseline_graph = build_pipeline_graph(baseline_op)
    ensemble_graph = build_pipeline_graph(ensemble_op)
    varied_graph = build_pipeline_graph(varied_op)

    # Initialize OperatorGraphRunner
    runner = OperatorGraphRunner(max_workers=args.max_workers)

    # Helper for scoring a single entry
    def score_single_entry(index: int, entry: DatasetEntry) -> Tuple[int, int, int]:
        """
        Returns (baseline_correct, ensemble_correct, varied_correct) for a single MMLU entry.
        The 'correct_answer' is in entry.metadata["correct_answer"] (e.g. "A").
        """
        logging.debug(f"Scoring Entry {index}: {entry}")
        query = entry.query
        choices = entry.choices
        correct_answer = entry.metadata.get("correct_answer", "").upper()

        # Baseline pipeline
        baseline_out = runner.run(
            baseline_graph,
            {"query": query, "choices": choices}  # typed inputs for SingleModelBaseline
        )
        base_pred = baseline_out.final_answer.upper()
        baseline_correct = 1 if (base_pred == correct_answer) else 0

        # Ensemble pipeline
        ensemble_out = runner.run(
            ensemble_graph,
            {"query": query, "choices": choices}  # typed inputs for MultiModelEnsemble
        )
        ens_pred = ensemble_out.final_answer.upper()
        ensemble_correct = 1 if (ens_pred == correct_answer) else 0

        # Varied pipeline
        varied_out = runner.run(
            varied_graph,
            {"query": query, "choices": choices}  # typed inputs for VariedModelEnsemble
        )
        varied_pred = varied_out.final_answer.upper()
        varied_correct = 1 if (varied_pred == correct_answer) else 0

        return (baseline_correct, ensemble_correct, varied_correct)

    # Score all entries in parallel
    baseline_total = 0
    ensemble_total = 0
    varied_total = 0
    total_count = len(dataset_entries) or 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {
            executor.submit(score_single_entry, index, e): index
            for index, e in enumerate(dataset_entries)
        }
        for future in concurrent.futures.as_completed(future_map):
            b_correct, e_correct, v_correct = future.result()
            baseline_total += b_correct
            ensemble_total += e_correct
            varied_total += v_correct

    # Calculate accuracy & display final results
    baseline_acc = 100.0 * baseline_total / total_count
    ensemble_acc = 100.0 * ensemble_total / total_count
    varied_acc = 100.0 * varied_total / total_count

    table = PrettyTable()
    table.field_names = ["Pipeline", "Accuracy (%)", "Num Samples"]
    table.add_row(["SingleModelBaseline", f"{baseline_acc:.2f}", total_count])
    table.add_row(["MultiModelEnsemble (Judge aggregator)", f"{ensemble_acc:.2f}", total_count])
    table.add_row(["VariedModelEnsemble", f"{varied_acc:.2f}", total_count])

    print("\nFinal Results:")
    print(table)


if __name__ == "__main__":
    sys.exit(main())