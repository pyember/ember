"""
HaluEval Experiment – Binary Classification (Hallucinated vs. Not Hallucinated)

This script runs an end‐to‐end experiment using the HaluEval dataset. The HaluEval
prepper produces two dataset entries per original example (one with the correct answer
("Not Hallucinated", choice "A") and one with the hallucinated answer (choice "B")).
We then run three pipelines (a single-model baseline, a multi-model ensemble, and a varied-model ensemble)
to score each dataset entry and compare the predicted choice against the ground truth.

Usage:
    python halueval_experiment_example.py \
        --config_name="qa" --num_samples=5 --max_workers=4
"""

import argparse
import concurrent.futures
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel
from prettytable import PrettyTable

# ------------------------------------------------------------------------------
# Ember imports for model registry, LM modules, and pipelines.
from ember.core import (
    non,
)  # Provides operators such as Ensemble, GetAnswer, JudgeSynthesis, VariedEnsemble
from ember.core.app_context import get_ember_context
from ember.core.configs.config import initialize_system
from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig
from ember.examples.mcq_experiment_example import (
    EnsureValidChoiceOperator,
    SingleModelBaseline,
    MultiModelEnsemble,
    VariedModelEnsemble,
)

# ------------------------------------------------------------------------------
# For dataset loading (we use the HaluEval prepper)
from ember.core.utils.data.datasets_registry.halueval import HaluEvalConfig
from ember.core.utils.data.service import DatasetService
from ember.core.utils.data.metadata_registry import DatasetMetadataRegistry
from ember.core.utils.data.loader_factory import DatasetLoaderFactory
from ember.core.utils.data.base.loaders import HuggingFaceDatasetLoader
from ember.core.utils.data.base.samplers import DatasetSampler
from ember.core.utils.data.base.validators import DatasetValidator
from ember.core.utils.data.initialization import initialize_dataset_registry

# ------------------------------------------------------------------------------
# For constructing operator graphs (for concurrent execution)
from ember.xcs.graph.xcs_graph import XCSGraph  # our OperatorGraph equivalent

# ADD the execute_graph import:
from ember.xcs.engine.xcs_engine import execute_graph


###############################################################################
# Helper function: build a one-node execution plan (graph) wrapping a pipeline operator.
###############################################################################
def build_pipeline_graph(pipeline_op: non.Operator) -> XCSGraph:
    graph: XCSGraph = XCSGraph()
    graph.add_node(operator=pipeline_op, node_id=pipeline_op.name)
    return graph


###############################################################################
# Command-line argument parsing
###############################################################################
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HaluEval Experiment – Evaluate binary classification (Hallucinated vs. Not Hallucinated)"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="qa",
        help="HaluEval sub-config name (default 'qa').",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of HaluEval samples to test on.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Max number of threads for parallel scoring (default: auto).",
    )
    return parser.parse_args()


###############################################################################
# Main experiment code
###############################################################################
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_arguments()

    # Initialize the system using the context's registry.
    context = get_ember_context()
    initialize_system(registry=context.registry)
    logging.info("Initialized model registry from context.")

    # (Optional) Setup API keys from environment if needed.
    # e.g. for OpenAI: os.environ.get("OPENAI_API_KEY")

    # --------------------------------------------------------------------------
    # Initialize dataset registry and loader factory for HaluEval
    metadata_registry: DatasetMetadataRegistry = DatasetMetadataRegistry()
    loader_factory: DatasetLoaderFactory = DatasetLoaderFactory()
    initialize_dataset_registry(metadata_registry, loader_factory)
    # (Assuming that HaluEval is registered in the metadata registry with name "halueval")

    halueval_info = metadata_registry.get("halueval")
    if halueval_info is None:
        logging.error(
            "HaluEval dataset not found in metadata registry. Check registration."
        )
        sys.exit(1)
    prepper_cls = loader_factory.get_prepper_class("halueval")
    if prepper_cls is None:
        logging.error("No HaluEval prepper found in the loader factory.")
        sys.exit(1)

    halueval_config: HaluEvalConfig = HaluEvalConfig(
        config_name=args.config_name, split="data"
    )
    prepper = prepper_cls(config=halueval_config)

    # Instantiate dataset service and load entries.
    dataset_service = DatasetService(
        loader=HuggingFaceDatasetLoader(),
        validator=DatasetValidator(),
        sampler=DatasetSampler(),
        transformers=[],  # No additional transformers
    )
    dataset_entries: List[Any] = dataset_service.load_and_prepare(
        dataset_info=halueval_info,
        prepper=prepper,
        config=halueval_config,
        num_samples=args.num_samples,
    )
    logging.info(f"Loaded {len(dataset_entries)} HaluEval dataset entries.")

    # --------------------------------------------------------------------------
    # Build three pipeline operators for comparison.
    # We use:
    #   1) SingleModelBaseline – one-LM baseline.
    #   2) MultiModelEnsemble – three-LM ensemble with Judge synthesis.
    #   3) VariedModelEnsemble – ensemble with varied LMModule configurations.
    #
    # Note: For HaluEval, the query is pre-rendered by the HaluEvalPrepper into a text that
    # asks whether the candidate answer is supported by the provided knowledge.
    #
    # The choices are expected to be: {"A": "Not Hallucinated", "B": "Hallucinated"}
    # and the ground truth is stored in metadata["correct_answer"].

    baseline_op = SingleModelBaseline(model_name="openai:gpt-4o", temperature=0.0)
    ensemble_op = MultiModelEnsemble(model_name="openai:gpt-4o", temperature=0.7)
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

    # Wrap each pipeline in a one-node execution graph.
    baseline_graph: XCSGraph = build_pipeline_graph(pipeline_op=baseline_op)
    ensemble_graph: XCSGraph = build_pipeline_graph(pipeline_op=ensemble_op)
    varied_graph: XCSGraph = build_pipeline_graph(pipeline_op=varied_op)

    # --------------------------------------------------------------------------
    # Define a scoring function to evaluate one dataset entry using all pipelines.
    def score_entry(index: int, entry: Any) -> Tuple[int, int, int]:
        """
        Scores a single dataset entry using the three pipelines.
        The ground truth (correct answer) is in entry.metadata["correct_answer"].
        Returns a tuple of 0/1 values for (baseline, ensemble, varied) accuracy.
        """
        query: str = entry.query
        choices: Dict[str, str] = entry.choices
        correct_answer: str = entry.metadata.get("correct_answer", "").upper()

        # Run baseline pipeline with execute_graph:
        baseline_out = execute_graph(
            graph=baseline_graph,
            global_input={"query": query, "choices": choices},
            max_workers=args.max_workers,
        )
        base_pred: str = baseline_out.final_answer.upper()
        baseline_correct: int = 1 if base_pred == correct_answer else 0

        # Run ensemble pipeline with execute_graph:
        ensemble_out = execute_graph(
            graph=ensemble_graph,
            global_input={"query": query, "choices": choices},
            max_workers=args.max_workers,
        )
        ens_pred: str = ensemble_out.final_answer.upper()
        ensemble_correct: int = 1 if ens_pred == correct_answer else 0

        # Run varied pipeline with execute_graph:
        varied_out = execute_graph(
            graph=varied_graph,
            global_input={"query": query, "choices": choices},
            max_workers=args.max_workers,
        )
        varied_pred: str = varied_out.final_answer.upper()
        varied_correct: int = 1 if varied_pred == correct_answer else 0

        return (baseline_correct, ensemble_correct, varied_correct)

    # --------------------------------------------------------------------------
    # Score all dataset entries in parallel.
    total_entries: int = len(dataset_entries) or 1
    baseline_sum: int = 0
    ensemble_sum: int = 0
    varied_sum: int = 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        future_to_index = {
            executor.submit(score_entry, index=i, entry=entry): i
            for i, entry in enumerate(dataset_entries)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            b, e, v = future.result()
            baseline_sum += b
            ensemble_sum += e
            varied_sum += v

    baseline_acc: float = 100.0 * baseline_sum / total_entries
    ensemble_acc: float = 100.0 * ensemble_sum / total_entries
    varied_acc: float = 100.0 * varied_sum / total_entries

    result_table: PrettyTable = PrettyTable()
    result_table.field_names = ["Pipeline", "Accuracy (%)", "Num Samples"]
    result_table.add_row(["SingleModelBaseline", f"{baseline_acc:.2f}", total_entries])
    result_table.add_row(["MultiModelEnsemble", f"{ensemble_acc:.2f}", total_entries])
    result_table.add_row(["VariedModelEnsemble", f"{varied_acc:.2f}", total_entries])

    print("\nFinal Results:")
    print(result_table)


if __name__ == "__main__":
    sys.exit(main())
