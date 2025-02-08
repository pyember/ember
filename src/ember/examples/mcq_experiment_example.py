"""
MCQ Experiment (MMLU) with parallel per-question scoring.

This module defines three pipelines for evaluating multiple-choice questions using
language models. The pipelines are:

1. SingleModelBaseline:
   - non.Ensemble(num_units=1) → non.GetAnswer → EnsureValidChoiceOperator
2. MultiModelEnsemble:
   - non.Ensemble(num_units=3) → non.JudgeSynthesis → non.GetAnswer → EnsureValidChoiceOperator
3. VariedModelEnsemble:
   - non.VariedEnsemble → non.JudgeSynthesis → non.GetAnswer → EnsureValidChoiceOperator

Usage:
    python -m src.ember.examples.mcq_experiment_example \
        --config_name="abstract_algebra" --num_samples=5 --max_workers=4
"""

import argparse
import concurrent.futures
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel
from prettytable import PrettyTable

# ember imports: use only the typed pipeline definitions (avoid direct registry references).
from ember.xcs.non import Ensemble, JudgeSynthesis
from ember.registry.model.core.model_registry import get_model_registry
from ember.xcs import non
from ember.xcs.graph_ir.operator_graph import OperatorGraph
from ember.xcs.graph_ir.operator_graph_runner import OperatorGraphRunner
from ember.registry.model.core.model_registry import GLOBAL_MODEL_REGISTRY
from ember.xcs.configs.config import CONFIG, initialize_system

# For dataset usage:
from ember.registry.dataset.base.models import DatasetEntry
from ember.registry.dataset.datasets.mmlu import MMLUConfig
from ember.registry.dataset.registry.service import DatasetService
from ember.registry.dataset.registry.metadata_registry import DatasetMetadataRegistry
from ember.registry.dataset.registry.loader_factory import DatasetLoaderFactory
from ember.registry.dataset.base.loaders import HuggingFaceDatasetLoader
from ember.registry.dataset.base.samplers import DatasetSampler
from ember.registry.dataset.base.validators import DatasetValidator
from ember.registry.dataset.registry.initialization import initialize_dataset_registry
from ember.xcs.scheduler import ExecutionPlan
from ember.src.ember.registry.operator.core.operator_base import (
    LMModule,
    Operator,
    OperatorMetadata,
    OperatorType,
)
from ember.registry.prompt_signature.signatures import Signature
from ember.registry.operator.modules.lm_modules import LMModuleConfig


###############################################################################
# A) EnsureValidChoiceOperator
###############################################################################
class EnsureValidChoiceInputs(BaseModel):
    """Input data for the EnsureValidChoiceOperator."""

    query: str
    partial_answer: str
    choices: Dict[str, str]


class EnsureValidChoiceSignature(Signature):
    """Signature for the EnsureValidChoiceOperator."""

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
    """Operator that refines or validates a partial answer to ensure it is a valid choice.

    This operator attempts to convert a partial or invalid answer into a valid choice key
    using a language model (LM) fallback mechanism.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="ENSURE_VALID_CHOICE",
        description=(
            "Refines or validates that final_answer is one of the valid choices, "
            "or tries to fix it via LM."
        ),
        operator_type=OperatorType.RECURRENT,
        signature=EnsureValidChoiceSignature(),
    )

    def __init__(
        self,
        model_name: str = "openai:o1",
        temperature: float = 0.0,
        max_tokens: Optional[int] = 16,
        max_retries: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initializes the EnsureValidChoiceOperator.

        Args:
            model_name: The identifier of the language model.
            temperature: Temperature setting for the language model.
            max_tokens: The maximum token count for LM outputs.
            max_retries: Maximum number of LM fallback attempts.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="EnsureValidChoiceOperator", signature=self.metadata.signature
        )
        lm_config: LMModuleConfig = LMModuleConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.lm_modules: List[LMModule] = [LMModule(config=lm_config)]
        self._max_retries: int = max_retries

    def forward(self, inputs: EnsureValidChoiceInputs) -> Dict[str, Any]:
        """Processes the input to ensure a valid choice is returned.

        Args:
            inputs: The input data containing the query, partial answer, and valid choices.

        Returns:
            A dictionary containing the key 'final_answer' with a valid choice.

        Raises:
            ValueError: If a valid answer cannot be obtained within the retry limit.
        """
        if inputs.partial_answer in inputs.choices:
            return {"final_answer": inputs.partial_answer}

        logging.info(
            f"[EnsureValidChoiceOperator] Invalid choice '{inputs.partial_answer}'. "
            f"Valid keys: {list(inputs.choices.keys())}. Attempting fallback..."
        )

        refined_answer: str = inputs.partial_answer
        for attempt_i in range(self._max_retries):
            prompt_context: Dict[str, str] = {
                "query": inputs.query,
                "partial_answer": refined_answer,
                "choices": "\n".join(f"{k}: {v}" for k, v in inputs.choices.items()),
            }
            prompt_str: str = self.build_prompt(prompt_context)

            fallback_output: str = (
                self.call_lm(prompt_str, self.lm_modules[0]).strip().upper()
            )
            logging.info(
                f"Attempt {attempt_i + 1}/{self._max_retries}, LM suggested: '{fallback_output}'"
            )

            if fallback_output in inputs.choices:
                return {"final_answer": fallback_output}
            refined_answer = fallback_output

        raise ValueError(
            f"[EnsureValidChoiceOperator] Could not map answer '{inputs.partial_answer}' "
            f"to any valid choice {list(inputs.choices.keys())} after {self._max_retries} attempts."
        )

    def to_plan(self, inputs: EnsureValidChoiceInputs) -> Optional[ExecutionPlan]:
        """Generates an execution plan if applicable.

        Args:
            inputs: The input data.

        Returns:
            An ExecutionPlan object if a plan can be generated, else None.
        """
        return None


###############################################################################
# 1) SingleModelBaseline pipeline
###############################################################################
class SingleModelBaselineInputs(BaseModel):
    """Input data for the SingleModelBaseline pipeline."""

    query: str
    choices: Dict[str, str]


class SingleModelBaselineOutputs(BaseModel):
    """Output data for the SingleModelBaseline pipeline."""

    final_answer: str


class SingleModelBaselineSignature(Signature):
    """Signature for the SingleModelBaseline pipeline."""

    required_inputs: List[str] = ["query", "choices"]
    input_model: Type[BaseModel] = SingleModelBaselineInputs


class SingleModelBaseline(
    non.Operator[SingleModelBaselineInputs, SingleModelBaselineOutputs]
):
    """One-model baseline pipeline that uses a single language model for answer extraction.

    Pipeline steps:
      1. non.Ensemble with one unit.
      2. non.GetAnswer to retrieve an answer.
      3. EnsureValidChoiceOperator to validate the answer.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="SINGLE_MODEL_BASELINE",
        description="One-LM baseline with EnsureValidChoice.",
        operator_type=OperatorType.RECURRENT,
        signature=SingleModelBaselineSignature(),
    )

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
    ) -> None:
        """Initializes the SingleModelBaseline pipeline.

        Args:
            model_name: The identifier for the language model.
            temperature: Temperature setting for the language model.
        """
        super().__init__(name="SingleModelBaseline", signature=self.metadata.signature)
        # Step 1: Create an ensemble operator with a single unit.
        self.ensemble = non.Ensemble(
            num_units=1, model_name=model_name, temperature=temperature
        )
        # Step 2: Create the get_answer operator.
        self.get_answer = non.GetAnswer(model_name=model_name, temperature=temperature)
        # Step 3: Create the ensure valid choice operator.
        self.ensure_valid_choice = EnsureValidChoiceOperator(
            model_name=model_name,
            temperature=0.0,
            max_tokens=16,
            max_retries=1,
        )

    def forward(self, inputs: SingleModelBaselineInputs) -> SingleModelBaselineOutputs:
        """Executes the SingleModelBaseline pipeline.

        Args:
            inputs: Pipeline inputs with the query and corresponding choices.

        Returns:
            An instance of SingleModelBaselineOutputs containing the validated final answer.
        """
        ensemble_output: Dict[str, Any] = self.ensemble(inputs={"query": inputs.query})
        responses: List[Any] = ensemble_output.get("responses", [])

        get_answer_output: Dict[str, Any] = self.get_answer(
            inputs={"query": inputs.query, "responses": responses}
        )
        intermediate_answer: str = get_answer_output.get("final_answer", "")

        valid_choice_output: Dict[str, Any] = self.ensure_valid_choice(
            inputs={
                "query": inputs.query,
                "partial_answer": intermediate_answer,
                "choices": inputs.choices,
            }
        )
        final_answer: str = valid_choice_output.get("final_answer", "")
        return SingleModelBaselineOutputs(final_answer=final_answer)


###############################################################################
# 2) MultiModelEnsemble pipeline
###############################################################################
class MultiModelEnsembleInputs(BaseModel):
    """Input data for the MultiModelEnsemble pipeline."""

    query: str
    choices: Dict[str, str]


class MultiModelEnsembleOutputs(BaseModel):
    """Output data for the MultiModelEnsemble pipeline."""

    final_answer: str


class MultiModelEnsembleSignature(Signature):
    """Signature for the MultiModelEnsemble pipeline."""

    required_inputs: List[str] = ["query", "choices"]
    input_model: Type[BaseModel] = MultiModelEnsembleInputs


class MultiModelEnsemble(
    non.Operator[MultiModelEnsembleInputs, MultiModelEnsembleOutputs]
):
    """Multi-model ensemble pipeline that aggregates multiple LM responses.

    Pipeline steps:
      1. non.Ensemble with three units.
      2. non.JudgeSynthesis to aggregate responses.
      3. non.GetAnswer for optional post-processing.
      4. EnsureValidChoiceOperator to verify the final answer.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="MULTI_MODEL_ENSEMBLE",
        description="Multi-model ensemble aggregator with judge step.",
        operator_type=OperatorType.RECURRENT,
        signature=MultiModelEnsembleSignature(),
    )

    def __init__(
        self,
        model_name: str = "claude-3.5-sonnet-latest",
        temperature: float = 0.7,
    ) -> None:
        """Initializes the MultiModelEnsemble pipeline.

        Args:
            model_name: The identifier for the language model.
            temperature: Temperature setting for the language model.
        """
        super().__init__(name="MultiModelEnsemble", signature=self.metadata.signature)
        # Step 1: Ensemble operator with three units.
        self.ensemble = non.Ensemble(
            num_units=3, model_name=model_name, temperature=temperature
        )
        # Step 2: Judge synthesis aggregator.
        self.judge = non.JudgeSynthesis(model_name=model_name, temperature=temperature)
        # Step 3: Optional get_answer operator.
        self.get_answer = non.GetAnswer(model_name=model_name, temperature=0.1)
        # Step 4: Ensure the final answer is valid.
        self.ensure_valid_choice = EnsureValidChoiceOperator(
            model_name=model_name, temperature=0.1, max_tokens=16, max_retries=1
        )

    def forward(self, inputs: MultiModelEnsembleInputs) -> MultiModelEnsembleOutputs:
        """Executes the MultiModelEnsemble pipeline.

        Args:
            inputs: Pipeline inputs with the query and valid choices.

        Returns:
            An instance of MultiModelEnsembleOutputs containing the aggregated final answer.
        """
        ensemble_output: Dict[str, Any] = self.ensemble(inputs={"query": inputs.query})
        responses: List[Any] = ensemble_output.get("responses", [])

        judge_output = self.judge(
            inputs={"query": inputs.query, "responses": responses}
        )
        judge_answer: str = getattr(judge_output, "final_answer", "")

        get_answer_output: Dict[str, Any] = self.get_answer(
            inputs={"query": inputs.query, "responses": [judge_answer]}
        )
        intermediate_answer: str = get_answer_output.get("final_answer", "")

        valid_choice_output: Dict[str, Any] = self.ensure_valid_choice(
            inputs={
                "query": inputs.query,
                "partial_answer": intermediate_answer,
                "choices": inputs.choices,
            }
        )
        final_answer: str = valid_choice_output.get("final_answer", "")
        return MultiModelEnsembleOutputs(final_answer=final_answer)


###############################################################################
# 2b) VariedModelEnsemble pipeline (refactored to match MultiModelEnsemble)
###############################################################################
class VariedModelEnsembleInputs(BaseModel):
    """Input data for the VariedModelEnsemble pipeline."""

    query: str
    choices: Dict[str, str]


class VariedModelEnsembleOutputs(BaseModel):
    """Output data for the VariedModelEnsemble pipeline."""

    final_answer: str


class VariedModelEnsembleSignature(Signature):
    """Signature for the VariedModelEnsemble pipeline."""

    required_inputs: List[str] = ["query", "choices"]
    input_model: Type[BaseModel] = VariedModelEnsembleInputs


class VariedModelEnsemble(
    non.Operator[VariedModelEnsembleInputs, VariedModelEnsembleOutputs]
):
    """Varied model ensemble pipeline using multiple language model configurations.

    Pipeline steps:
      1. non.VariedEnsemble to retrieve responses from varied models.
      2. non.JudgeSynthesis to synthesize an aggregated answer.
      3. non.GetAnswer for optional post-processing.
      4. EnsureValidChoiceOperator to validate the final answer.
    """

    metadata: OperatorMetadata = OperatorMetadata(
        code="VARIED_MODEL_ENSEMBLE",
        description=(
            "Multi-model pipeline aggregator with judge step, using VariedEnsemble for step #1."
        ),
        operator_type=OperatorType.RECURRENT,
        signature=VariedModelEnsembleSignature(),
    )

    def __init__(self, model_configs: List[LMModuleConfig]) -> None:
        """Initializes the VariedModelEnsemble pipeline.

        Args:
            model_configs: A list of LMModuleConfig objects defining the model configurations.
        """
        super().__init__(name="VariedModelEnsemble", signature=self.metadata.signature)
        # Step 1: Use VariedEnsemble with multiple model configurations.
        self.ensemble = non.VariedEnsemble(model_configs=model_configs)

        # Configure aggregator model parameters.
        aggregator_model_name: str = "openai:o1"
        aggregator_temp: float = 0.7

        self.judge = non.JudgeSynthesis(
            model_name=aggregator_model_name, temperature=aggregator_temp
        )
        self.get_answer = non.GetAnswer(
            model_name=aggregator_model_name,
            temperature=max(0.0, aggregator_temp - 0.6),
        )
        self.ensure_valid_choice = EnsureValidChoiceOperator(
            model_name=aggregator_model_name,
            temperature=aggregator_temp,
            max_tokens=16,
            max_retries=1,
        )

    def forward(self, inputs: VariedModelEnsembleInputs) -> VariedModelEnsembleOutputs:
        """Executes the VariedModelEnsemble pipeline.

        Args:
            inputs: Pipeline inputs with the query and valid choices.

        Returns:
            An instance of VariedModelEnsembleOutputs containing the final answer.
        """
        ensemble_output = self.ensemble(inputs={"query": inputs.query})
        responses: List[Any] = getattr(ensemble_output, "responses", [])

        judge_output = self.judge(
            inputs={"query": inputs.query, "responses": responses}
        )
        judge_answer: str = getattr(judge_output, "final_answer", "")

        get_answer_output: Dict[str, Any] = self.get_answer(
            inputs={"query": inputs.query, "responses": [judge_answer]}
        )
        intermediate_answer: str = get_answer_output.get("final_answer", "")

        valid_choice_output: Dict[str, Any] = self.ensure_valid_choice(
            inputs={
                "query": inputs.query,
                "partial_answer": intermediate_answer,
                "choices": inputs.choices,
            }
        )
        final_answer: str = valid_choice_output.get("final_answer", "")
        return VariedModelEnsembleOutputs(final_answer=final_answer)


###############################################################################
# 3) Build each pipeline as a one-node OperatorGraph
###############################################################################
def build_pipeline_graph(pipeline_op: non.Operator) -> OperatorGraph:
    """Wraps a single pipeline operator in an OperatorGraph.

    Args:
        pipeline_op: The pipeline operator to be wrapped.

    Returns:
        An OperatorGraph instance containing the given pipeline as a node.
    """
    graph: OperatorGraph = OperatorGraph()
    graph.add_node(pipeline_op, node_id=pipeline_op.name)
    return graph


###############################################################################
# 4) Main experiment code
###############################################################################
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="MCQ Experiment with Baseline vs. Multi-Model Ensemble."
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="abstract_algebra",
        help="MMLU sub-config (e.g. 'abstract_algebra').",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of MMLU samples to test on.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of threads for parallel scoring. Defaults to Python's choice.",
    )
    return parser.parse_args()


def setup_openai_api_key() -> None:
    """Sets up the OpenAI API key from environment variables if available."""
    api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
    if api_key:
        CONFIG.set("models", "openai_api_key", api_key)
        logging.info("OpenAI API key set from environment.")
    else:
        logging.warning("No OPENAI_API_KEY found; continuing without it.")


def main() -> None:
    """Main function to run the MCQ experiment."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args: argparse.Namespace = parse_arguments()

    # Initialize the system with the global model registry.
    initialize_system(registry=GLOBAL_MODEL_REGISTRY)
    setup_openai_api_key()

    # Initialize the MMLU dataset environment.
    metadata_registry: DatasetMetadataRegistry = DatasetMetadataRegistry()
    loader_factory: DatasetLoaderFactory = DatasetLoaderFactory()
    initialize_dataset_registry(metadata_registry, loader_factory)

    # Obtain MMLU dataset info and create a prepper.
    mmlu_info: Optional[Any] = metadata_registry.get("mmlu")
    if mmlu_info is None:
        logging.error("MMLU dataset not found. Ensure it is registered.")
        sys.exit(1)
    prepper_cls: Optional[Type] = loader_factory.get_prepper_class("mmlu")
    if prepper_cls is None:
        logging.error("No MMLU prepper found in the registry.")
        sys.exit(1)

    mmlu_config: MMLUConfig = MMLUConfig(config_name=args.config_name, split="dev")
    prepper = prepper_cls(config=mmlu_config)

    dataset_service: DatasetService = DatasetService(
        loader=HuggingFaceDatasetLoader(),
        validator=DatasetValidator(),
        sampler=DatasetSampler(),
        transformers=[],
    )
    dataset_entries: List[DatasetEntry] = dataset_service.load_and_prepare(
        dataset_info=mmlu_info,
        prepper=prepper,
        config=mmlu_config,
        num_samples=args.num_samples,
    )
    logging.info(
        f"Loaded {len(dataset_entries)} MMLU items (config={args.config_name})."
    )

    # Build the pipeline operators using the typed wrappers.
    baseline_op = SingleModelBaseline(model_name="gpt-4o", temperature=0.0)
    ensemble_op = MultiModelEnsemble(model_name="gpt-4o", temperature=0.7)
    varied_op = VariedModelEnsemble(
        model_configs=[
            LMModuleConfig(model_name="openai:gpt-4o", temperature=0.6),
            LMModuleConfig(
                model_name="anthropic:claude-3.5-sonnet-latest", temperature=0.8
            ),
            LMModuleConfig(model_name="openai:o1", temperature=0.4),
            LMModuleConfig(model_name="google:gemini-1.5-pro", temperature=0.5),
        ]
    )

    baseline_graph: OperatorGraph = build_pipeline_graph(pipeline_op=baseline_op)
    ensemble_graph: OperatorGraph = build_pipeline_graph(pipeline_op=ensemble_op)
    varied_graph: OperatorGraph = build_pipeline_graph(pipeline_op=varied_op)

    runner: OperatorGraphRunner = OperatorGraphRunner(max_workers=args.max_workers)

    def score_single_entry(index: int, entry: DatasetEntry) -> Tuple[int, int, int]:
        """Scores a single dataset entry using all three pipelines.

        Args:
            index: The index of the dataset entry.
            entry: The dataset entry to be scored.

        Returns:
            A tuple of integers (baseline_correct, ensemble_correct, varied_correct),
            where each value is 1 if the corresponding pipeline's answer matches the correct answer, otherwise 0.
        """
        logging.debug(f"Scoring Entry {index}: {entry}")
        query: str = entry.query
        choices: Dict[str, str] = entry.choices
        correct_answer: str = entry.metadata.get("correct_answer", "").upper()

        # Evaluate the baseline pipeline.
        baseline_out = runner.run(
            pipeline=baseline_graph,
            inputs={"query": query, "choices": choices},
        )
        base_pred: str = getattr(baseline_out, "final_answer", "").upper()
        baseline_correct: int = 1 if (base_pred == correct_answer) else 0

        # Evaluate the ensemble pipeline.
        ensemble_out = runner.run(
            pipeline=ensemble_graph,
            inputs={"query": query, "choices": choices},
        )
        ens_pred: str = getattr(ensemble_out, "final_answer", "").upper()
        ensemble_correct: int = 1 if (ens_pred == correct_answer) else 0

        # Evaluate the varied pipeline.
        varied_out = runner.run(
            pipeline=varied_graph,
            inputs={"query": query, "choices": choices},
        )
        varied_pred: str = getattr(varied_out, "final_answer", "").upper()
        varied_correct: int = 1 if (varied_pred == correct_answer) else 0

        return (baseline_correct, ensemble_correct, varied_correct)

    baseline_total: int = 0
    ensemble_total: int = 0
    varied_total: int = 0
    total_count: int = len(dataset_entries) or 1

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        future_map = {
            executor.submit(score_single_entry, index=index, entry=entry): index
            for index, entry in enumerate(dataset_entries)
        }
        for future in concurrent.futures.as_completed(future_map):
            b_correct, e_correct, v_correct = future.result()
            baseline_total += b_correct
            ensemble_total += e_correct
            varied_total += v_correct

    baseline_acc: float = 100.0 * baseline_total / total_count
    ensemble_acc: float = 100.0 * ensemble_total / total_count
    varied_acc: float = 100.0 * varied_total / total_count

    result_table: PrettyTable = PrettyTable()
    result_table.field_names = ["Pipeline", "Accuracy (%)", "Num Samples"]
    result_table.add_row(["SingleModelBaseline", f"{baseline_acc:.2f}", total_count])
    result_table.add_row(
        ["MultiModelEnsemble (Judge aggregator)", f"{ensemble_acc:.2f}", total_count]
    )
    result_table.add_row(["VariedModelEnsemble", f"{varied_acc:.2f}", total_count])

    print("\nFinal Results:")
    print(result_table)


if __name__ == "__main__":
    sys.exit(main())
