"""
MCQ Experiment (MMLU) with parallel per-question scoring.

This module defines three pipelines for evaluating multiple-choice questions using
language models. The pipelines are:

1. SingleModelBaseline:
   - non.Ensemble(num_units=1) → EnsureValidChoiceOperator
2. MultiModelEnsemble:
   - non.Ensemble(num_units=3) → non.JudgeSynthesis → EnsureValidChoiceOperator
3. VariedModelEnsemble:
   - non.VariedEnsemble → non.JudgeSynthesis → EnsureValidChoiceOperator

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
from ember.core.non import UniformEnsemble, JudgeSynthesis, VariedEnsemble
from ember.core.app_context import get_ember_context
from ember.core import non
from ember.core.configs.config import CONFIG, initialize_system

# For dataset usage:
from ember.core.utils.data.base.models import DatasetEntry
from ember.core.utils.data.datasets_registry.mmlu import MMLUConfig
from ember.core.utils.data.service import DatasetService
from ember.core.utils.data.metadata_registry import DatasetMetadataRegistry
from ember.core.utils.data.loader_factory import DatasetLoaderFactory
from ember.core.utils.data.base.loaders import HuggingFaceDatasetLoader
from ember.core.utils.data.base.samplers import DatasetSampler
from ember.core.utils.data.base.validators import DatasetValidator
from ember.core.utils.data.initialization import initialize_dataset_registry
from ember.xcs.scheduler import ExecutionPlan
from ember.core.registry.operator.base.operator_base import (
    LMModule,
    Operator,
    OperatorMetadata,
)
from ember.core.registry.prompt_specification.specification import Specification
from ember.core.registry.model.model_module.lm import LMModuleConfig

# Import from XCS engine:
from ember.xcs.engine.xcs_engine import execute_graph
from ember.xcs.graph.xcs_graph import OperatorGraph


###############################################################################
# A) EnsureValidChoiceOperator
###############################################################################
class EnsureValidChoiceInputs(BaseModel):
    """Input data for the EnsureValidChoiceOperator."""

    query: str
    partial_answer: str
    choices: Dict[str, str]


class EnsureValidChoiceSpecification(Specification):
    """Specification for the EnsureValidChoiceOperator."""

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
        description="Refines or validates that final_answer is one of the valid choices, or tries to fix it via LM.",
        specification=EnsureValidChoiceSpecification(),
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
            name="EnsureValidChoiceOperator", specification=self.metadata.specification
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


class SingleModelBaselineSpecification(Specification):
    """Specification for the SingleModelBaseline pipeline."""

    input_model: Type[BaseModel] = SingleModelBaselineInputs


class SingleModelBaseline(
    non.Operator[SingleModelBaselineInputs, SingleModelBaselineOutputs]
):
    """One-model baseline pipeline that uses a single language model for answer extraction.

    Pipeline steps:
      1. non.UniformEnsemble with one unit to generate an answer.
      2. EnsureValidChoiceOperator to validate the answer.
    """

    # Declare class variables
    specification: ClassVar[Specification] = SingleModelBaselineSpecification()
    metadata: ClassVar[OperatorMetadata] = OperatorMetadata(
        code="SINGLE_MODEL_BASELINE",
        description="One-LM baseline with EnsureValidChoice.",
        specification=specification,
    )

    # Declare instance variables
    model_name: str
    temperature: float
    ensemble: non.UniformEnsemble
    ensure_valid_choice: EnsureValidChoiceOperator

    def __init__(
        self,
        *,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
    ) -> None:
        """Initializes the SingleModelBaseline pipeline.

        Args:
            model_name: The identifier for the language model.
            temperature: Temperature setting for the language model.
        """
        # Store instance configuration
        self.model_name = model_name
        self.temperature = temperature

        # Create component operators
        self.ensemble = non.UniformEnsemble(
            num_units=1, model_name=model_name, temperature=temperature
        )
        self.ensure_valid_choice = EnsureValidChoiceOperator(
            model_name=model_name,
            temperature=0.0,
            max_tokens=16,
            max_retries=1,
        )

    def forward(
        self, *, inputs: SingleModelBaselineInputs
    ) -> SingleModelBaselineOutputs:
        """Executes the SingleModelBaseline pipeline.

        Args:
            inputs: Pipeline inputs with the query and corresponding choices.

        Returns:
            An instance of SingleModelBaselineOutputs containing the validated final answer.
        """
        # Step 1: Generate responses using the ensemble
        ensemble_output = self.ensemble(query=inputs.query)
        responses = ensemble_output.responses

        # If we have a response, use the first one as our answer
        intermediate_answer = responses[0] if responses else ""

        # Step 2: Ensure the answer is valid
        valid_choice_output = self.ensure_valid_choice(
            query=inputs.query,
            partial_answer=intermediate_answer,
            choices=inputs.choices,
        )

        # Return the final answer
        return SingleModelBaselineOutputs(
            final_answer=valid_choice_output.get("final_answer", "")
        )


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


class MultiModelEnsembleSpecification(Specification):
    """Specification for the MultiModelEnsemble pipeline."""

    input_model: Type[BaseModel] = MultiModelEnsembleInputs


class MultiModelEnsemble(
    non.Operator[MultiModelEnsembleInputs, MultiModelEnsembleOutputs]
):
    """Multi-model ensemble pipeline that aggregates multiple LM responses.

    Pipeline steps:
      1. non.UniformEnsemble with three units.
      2. non.JudgeSynthesis to aggregate responses.
      3. EnsureValidChoiceOperator to verify the final answer.
    """

    # Declare class variables
    specification: ClassVar[Specification] = MultiModelEnsembleSpecification()
    metadata: ClassVar[OperatorMetadata] = OperatorMetadata(
        code="MULTI_MODEL_ENSEMBLE",
        description="Multi-model ensemble aggregator with judge step.",
        specification=specification,
    )

    # Declare instance variables
    model_name: str
    temperature: float
    ensemble: non.UniformEnsemble
    judge: non.JudgeSynthesis
    ensure_answer_format_validity: EnsureValidChoiceOperator

    def __init__(
        self,
        *,
        model_name: str = "claude-3.5-sonnet-latest",
        temperature: float = 0.7,
    ) -> None:
        """Initializes the MultiModelEnsemble pipeline.

        Args:
            model_name: The identifier for the language model.
            temperature: Temperature setting for the language model.
        """
        # Store instance configuration
        self.model_name = model_name
        self.temperature = temperature

        # Create component operators
        self.ensemble = non.UniformEnsemble(
            num_units=3, model_name=model_name, temperature=temperature
        )
        self.judge = non.JudgeSynthesis(model_name=model_name, temperature=temperature)
        self.ensure_answer_format_validity = EnsureValidChoiceOperator(
            model_name=model_name, temperature=0.1, max_tokens=16, max_retries=1
        )

    def forward(self, *, inputs: MultiModelEnsembleInputs) -> MultiModelEnsembleOutputs:
        """Executes the MultiModelEnsemble pipeline.

        Args:
            inputs: Pipeline inputs with the query and valid choices.

        Returns:
            An instance of MultiModelEnsembleOutputs containing the aggregated final answer.
        """
        # Step 1: Generate diverse responses using ensemble
        ensemble_output = self.ensemble(query=inputs.query)
        responses = ensemble_output.responses

        # Step 2: Synthesize responses using judge
        judge_output = self.judge(query=inputs.query, responses=responses)
        judge_answer = judge_output.final_answer

        # Step 3: Ensure the answer is valid
        valid_choice_output = self.ensure_answer_format_validity(
            query=inputs.query, partial_answer=judge_answer, choices=inputs.choices
        )

        # Return the final answer
        return MultiModelEnsembleOutputs(
            final_answer=valid_choice_output.get("final_answer", "")
        )


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


class VariedModelEnsembleSpecification(Specification):
    """Specification for the VariedModelEnsemble pipeline."""

    input_model: Type[BaseModel] = VariedModelEnsembleInputs


class VariedModelEnsemble(
    non.Operator[VariedModelEnsembleInputs, VariedModelEnsembleOutputs]
):
    """Varied model ensemble pipeline using multiple language model configurations.

    Pipeline steps:
      1. non.VariedEnsemble to retrieve responses from varied models.
      2. non.JudgeSynthesis to synthesize an aggregated answer.
      3. EnsureValidChoiceOperator to validate the final answer.
    """

    # Declare class variables
    specification: ClassVar[Specification] = VariedModelEnsembleSpecification()
    metadata: ClassVar[OperatorMetadata] = OperatorMetadata(
        code="VARIED_MODEL_ENSEMBLE",
        description=(
            "Multi-model pipeline aggregator with judge step, using VariedEnsemble for step #1."
        ),
        specification=specification,
    )

    # Declare instance variables
    model_configs: List[LMModuleConfig]
    ensemble: non.VariedEnsemble
    judge: non.JudgeSynthesis
    ensure_answer_format_validity: EnsureValidChoiceOperator
    aggregator_model_name: str
    aggregator_temp: float

    def __init__(self, *, model_configs: List[LMModuleConfig]) -> None:
        """Initializes the VariedModelEnsemble pipeline.

        Args:
            model_configs: A list of LMModuleConfig objects defining the model configurations.
        """
        # Store instance configuration
        self.model_configs = model_configs

        # Configure aggregator model parameters
        self.aggregator_model_name = "openai:o1"
        self.aggregator_temp = 0.7

        # Create component operators
        self.ensemble = non.VariedEnsemble(model_configs=model_configs)
        self.judge = non.JudgeSynthesis(
            model_name=self.aggregator_model_name, temperature=self.aggregator_temp
        )
        self.ensure_answer_format_validity = EnsureValidChoiceOperator(
            model_name=self.aggregator_model_name,
            temperature=self.aggregator_temp,
            max_tokens=16,
            max_retries=1,
        )

    def forward(
        self, *, inputs: VariedModelEnsembleInputs
    ) -> VariedModelEnsembleOutputs:
        """Executes the VariedModelEnsemble pipeline.

        Args:
            inputs: Pipeline inputs with the query and valid choices.

        Returns:
            An instance of VariedModelEnsembleOutputs containing the final answer.
        """
        # Step 1: Generate diverse responses from multiple models
        ensemble_output = self.ensemble(query=inputs.query)
        responses = ensemble_output.responses

        # Step 2: Synthesize responses using judge
        judge_output = self.judge(query=inputs.query, responses=responses)
        judge_answer = judge_output.final_answer

        # Step 3: Ensure the answer is valid
        valid_choice_output = self.ensure_answer_format_validity(
            query=inputs.query,
            partial_answer=judge_answer,
            choices=inputs.choices,
        )

        # Return the final answer
        return VariedModelEnsembleOutputs(
            final_answer=valid_choice_output.get("final_answer", "")
        )


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

    # Initialize the system with the context's registry.
    context = get_ember_context()
    initialize_system(registry=context.registry)
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

        # Evaluate the baseline pipeline:
        baseline_out = execute_graph(
            graph=baseline_graph,
            global_input={"query": query, "choices": choices},
            max_workers=args.max_workers,
        )
        base_pred: str = getattr(baseline_out, "final_answer", "").upper()
        baseline_correct: int = 1 if (base_pred == correct_answer) else 0

        # Evaluate the ensemble pipeline:
        ensemble_out = execute_graph(
            graph=ensemble_graph,
            global_input={"query": query, "choices": choices},
            max_workers=args.max_workers,
        )
        ens_pred: str = getattr(ensemble_out, "final_answer", "").upper()
        ensemble_correct: int = 1 if (ens_pred == correct_answer) else 0

        # Evaluate the varied pipeline:
        varied_out = execute_graph(
            graph=varied_graph,
            global_input={"query": query, "choices": choices},
            max_workers=args.max_workers,
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
