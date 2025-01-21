"""Base NoN package graph example on MCQ.

Usage:
    python non_package_graph_example.py [--num_questions NUM] [--dataset DATASET] [--dataset_config CONFIG]

Example:
    python non_package_graph_example.py --num_questions 10 --dataset mmlu --dataset_config abstract_algebra
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

from tqdm import tqdm

from avior.core.configs.config import set_config
from avior.core.non_graph import NoNGraph
from avior.registry import non
from avior.registry.dataset.dataset_registry import DatasetRegistry
from avior.registry.eval_function.eval_function_registry import (
    EvaluationResult,
    EvaluatorRegistry,
    MultipleChoiceEvaluator,
)
from avior.registry.model_registry.model_registry import ModelRegistry
from avior.registry.operators.operator_base import OperatorContext, NoNModule
from avior.registry.operators.operator_registry import OperatorRegistry
from example_architectures import nested_module_graph


# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def setup_openai_api_key() -> None:
    """Sets up the OpenAI API key from the environment variable.

    The function attempts to retrieve the OpenAI API key from the
    'OPENAI_API_KEY' environment variable. If found, it sets
    the key in the configuration. Otherwise, it prints a warning message.
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        set_config("models", "openai_api_key", openai_api_key)
    else:
        logging.info(
            "OPENAI_API_KEY_DELUXECHAT_RESEARCH not found in environment variables"
        )


def parse_arguments() -> argparse.Namespace:
    """Parses and returns the command-line arguments.

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run ensemble scoring on a subset of questions."
    )
    parser.add_argument(
        "--num_questions", type=int, default=10, help="Number of questions to use"
    )
    parser.add_argument("--dataset", type=str, default="mmlu", help="Dataset to use")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="high_school_mathematics",
        help="Dataset configuration",
    )
    return parser.parse_args()


def initialize_registries() -> (
    Tuple[ModelRegistry, OperatorRegistry, DatasetRegistry, EvaluatorRegistry]
):
    """Initializes and returns the necessary registries.

    Returns:
        A tuple containing instances of ModelRegistry, OperatorRegistry,
        DatasetRegistry, and EvaluatorRegistry.
    """
    logging.info("Initializing registries")
    model_registry = ModelRegistry()
    operator_registry = OperatorRegistry()
    dataset_registry = DatasetRegistry()
    evaluator_registry = EvaluatorRegistry()
    evaluator_registry.register("multiple_choice", MultipleChoiceEvaluator())
    logging.info("Registries initialized successfully")
    return model_registry, operator_registry, dataset_registry, evaluator_registry


def load_dataset(
    dataset_registry: DatasetRegistry, args: argparse.Namespace
) -> List[OperatorContext]:
    """Loads and prepares the dataset based on the provided arguments.

    Args:
        dataset_registry: An instance of DatasetRegistry.
        args: Parsed command-line arguments.

    Returns:
        A list of OperatorContext objects representing the selected questions.

    Raises:
        ValueError: If the specified dataset is not found in the registry.
    """
    logging.info(
        f"Getting formatted dataset: {args.dataset}, num_questions: {args.num_questions}"
    )
    dataset_info_and_loader = dataset_registry.get(args.dataset)
    if dataset_info_and_loader is None:
        raise ValueError(f"Dataset {args.dataset} not found in registry")
    _, dataset_loader_prepper = dataset_info_and_loader
    selected_questions = dataset_loader_prepper.load_and_prepare(
        num_questions=args.num_questions, subject=args.dataset_config
    )
    logging.info(f"Retrieved {len(selected_questions)} questions")
    return selected_questions


def define_non_graph() -> NoNModule:
    """Defining a nested NoN graph using NoNModules.

    Returns:
        An instance of NoNModule representing the nested network.
    """
    logging.info("Defining nested NoN graph using NoNModules")
    non_graph = nested_module_graph()
    logging.info("Nested NoN graph created")
    return non_graph


def score_single_question(
    question: OperatorContext,
    non_graph: NoNGraph,
    evaluator_registry: EvaluatorRegistry,
) -> Tuple[EvaluationResult, OperatorContext]:
    """Scores a single question using the NoN graph.

    Args:
        question: An OperatorContext object representing the question.
        non_graph: An instance of NoNGraph to process the question.
        evaluator_registry: An instance of EvaluatorRegistry for scoring.

    Returns:
        A tuple containing the EvaluationResult and the original OperatorContext.
    """
    logging.debug(f"Scoring question: {question.query[:50]}...")
    final_answer = non_graph.forward(question)
    evaluation_result = evaluator_registry.get("multiple_choice").score(
        final_answer, question.metadata["correct_answer"], choices=question.choices
    )
    logging.debug(f"Question scored. Result: {evaluation_result}")
    return evaluation_result, question


def score_non_graph(
    questions: List[OperatorContext],
    non_graph: NoNGraph,
    evaluator_registry: EvaluatorRegistry,
) -> float:
    """Scores the NoN graph on a list of questions.

    Args:
        questions: A list of OperatorContext objects representing the questions.
        non_graph: An instance of NoNGraph to process the questions.
        evaluator_registry: An instance of EvaluatorRegistry for scoring.

    Returns:
        The accuracy score as a float percentage.
    """
    logging.info("Starting score_non_graph function")
    total_score = 0
    with ThreadPoolExecutor() as executor:
        logging.info("Starting ThreadPoolExecutor for question scoring")
        future_to_question = {
            executor.submit(
                score_single_question, question, non_graph, evaluator_registry
            ): question
            for question in tqdm(questions)
        }
        for future in as_completed(future_to_question):
            evaluation_result, _ = future.result()
            total_score += evaluation_result.score
        logging.info("Finished scoring all questions")

    accuracy = (total_score / len(questions)) * 100
    logging.info(f"Calculated accuracy: {accuracy}%")
    return accuracy


def main() -> None:
    """Main function to run the MCQ experiment."""
    logging.info("Starting main function")

    setup_openai_api_key()
    args = parse_arguments()

    _, _, dataset_registry, evaluator_registry = initialize_registries()
    selected_questions = load_dataset(dataset_registry, args)
    non_graph = define_non_graph()

    logging.info("Starting to score NoN graph")
    score = score_non_graph(
        questions=selected_questions,
        non_graph=non_graph,
        evaluator_registry=evaluator_registry,
    )
    logging.info(f"Final Score: {score}")


if __name__ == "__main__":
    main()
