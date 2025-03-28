"""Benchmark models on specialized datasets (AIME, GPQA, Codeforces).

This example demonstrates how to:
1. Load and prepare specialized datasets
2. Evaluate multiple models on these datasets
3. Compare performance across different types of tasks
4. Generate performance reports

Usage:
    python -m ember.examples.advanced.model_benchmark_specialized_datasets [--dataset DATASET] [--samples N]
"""

import argparse
import logging
import re
import sys
import time
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_VISUALIZATION = True
except ImportError:
    logger.warning("Matplotlib not available. Visualizations will be skipped.")
    HAS_VISUALIZATION = False

from ember.api import datasets, models, DatasetBuilder
from ember.core.exceptions import GatedDatasetAuthenticationError
from ember.core.utils.eval.numeric_answer import AIMEAnswerEvaluator
from ember.core.utils.eval.evaluators import MultipleChoiceEvaluator
from ember.core.utils.eval.code_execution import CodeCompetitionEvaluator


def evaluate_aime(
    models_config: List[Tuple[str, Any]], sample_size: int = 5
) -> Dict[str, Any]:
    """Evaluate models on AIME dataset.

    Args:
        models_config: List of (name, model) tuples to evaluate
        sample_size: Number of problems to sample

    Returns:
        Results dictionary
    """
    logger.info("Evaluating models on AIME dataset...")

    try:
        # Load dataset
        aime_data = datasets("aime")
        problems = aime_data.sample(sample_size)

        if len(problems) == 0:
            logger.error("AIME dataset loaded but contains no problems")
            return {"success": False, "error": "Empty dataset"}

        # Initialize evaluator
        evaluator = AIMEAnswerEvaluator()

        # Track results
        results = {"success": True, "model_results": {}, "problems": []}

        # Run evaluation
        for i, problem in enumerate(problems):
            logger.info(f"\nProblem {i+1}: {problem.query[:100]}...")
            logger.info(f"Expected answer: {problem.metadata['correct_answer']}")

            problem_result = {
                "id": problem.metadata.get("problem_id", f"problem_{i}"),
                "query": problem.query,
                "answer": problem.metadata["correct_answer"],
                "model_responses": {},
            }

            # Evaluate each model
            for name, model in models_config:
                if name not in results["model_results"]:
                    results["model_results"][name] = {
                        "correct": 0,
                        "total": 0,
                        "time": 0,
                    }

                start_time = time.time()
                response = model(problem.query)
                inference_time = time.time() - start_time

                result = evaluator.evaluate(
                    response, problem.metadata["correct_answer"]
                )

                # Update results
                problem_result["model_responses"][name] = {
                    "response": response[:200] + "..."
                    if len(response) > 200
                    else response,
                    "is_correct": result.is_correct,
                    "extracted_value": result.metadata.get(
                        "extracted_value", "Not found"
                    ),
                    "method": result.metadata.get("extracted_method", "Unknown"),
                    "time": inference_time,
                }

                if result.is_correct:
                    results["model_results"][name]["correct"] += 1
                results["model_results"][name]["total"] += 1
                results["model_results"][name]["time"] += inference_time

                logger.info(
                    f"{name}: {'✓' if result.is_correct else '✗'} "
                    + f"(Found: {result.metadata.get('extracted_value', 'Not found')}) "
                    + f"[{inference_time:.2f}s]"
                )

            results["problems"].append(problem_result)

        # Calculate accuracy
        for name in results["model_results"]:
            model_data = results["model_results"][name]
            model_data["accuracy"] = (
                model_data["correct"] / model_data["total"]
                if model_data["total"] > 0
                else 0
            )
            model_data["avg_time"] = (
                model_data["time"] / model_data["total"]
                if model_data["total"] > 0
                else 0
            )

        return results

    except Exception as e:
        logger.error(f"AIME evaluation error: {e}")
        return {"success": False, "error": str(e)}


def evaluate_gpqa(
    models_config: List[Tuple[str, Any]], sample_size: int = 5
) -> Dict[str, Any]:
    """Evaluate models on GPQA dataset.

    Args:
        models_config: List of (name, model) tuples to evaluate
        sample_size: Number of problems to sample

    Returns:
        Results dictionary
    """
    logger.info("Evaluating models on GPQA dataset...")

    try:
        # Load dataset
        gpqa_data = datasets("gpqa")
        problems = gpqa_data.sample(sample_size)

        if len(problems) == 0:
            logger.error("GPQA dataset loaded but contains no questions")
            return {"success": False, "error": "Empty dataset"}

        # Initialize evaluator
        evaluator = MultipleChoiceEvaluator()

        # Track results
        results = {"success": True, "model_results": {}, "problems": []}

        # Run evaluation
        for i, problem in enumerate(problems):
            logger.info(f"\nQuestion {i+1}: {problem.query[:100]}...")
            logger.info(f"Choices: {list(problem.choices.keys())}")
            logger.info(f"Expected answer: {problem.metadata['correct_answer']}")

            # Format prompt
            prompt = problem.query + "\n\n"
            for key, choice in problem.choices.items():
                prompt += f"{key}. {choice}\n"

            problem_result = {
                "id": problem.metadata.get("id", f"question_{i}"),
                "query": problem.query,
                "choices": problem.choices,
                "answer": problem.metadata["correct_answer"],
                "subject": problem.metadata.get("subject", "Unknown"),
                "model_responses": {},
            }

            # Evaluate each model
            for name, model in models_config:
                if name not in results["model_results"]:
                    results["model_results"][name] = {
                        "correct": 0,
                        "total": 0,
                        "time": 0,
                    }

                start_time = time.time()
                response = model(prompt)
                inference_time = time.time() - start_time

                result = evaluator.evaluate(
                    response, problem.metadata["correct_answer"]
                )

                # Update results
                problem_result["model_responses"][name] = {
                    "response": response[:200] + "..."
                    if len(response) > 200
                    else response,
                    "is_correct": result.is_correct,
                    "time": inference_time,
                }

                if result.is_correct:
                    results["model_results"][name]["correct"] += 1
                results["model_results"][name]["total"] += 1
                results["model_results"][name]["time"] += inference_time

                logger.info(
                    f"{name}: {'✓' if result.is_correct else '✗'} [{inference_time:.2f}s]"
                )

            results["problems"].append(problem_result)

        # Calculate accuracy
        for name in results["model_results"]:
            model_data = results["model_results"][name]
            model_data["accuracy"] = (
                model_data["correct"] / model_data["total"]
                if model_data["total"] > 0
                else 0
            )
            model_data["avg_time"] = (
                model_data["time"] / model_data["total"]
                if model_data["total"] > 0
                else 0
            )

        return results

    except GatedDatasetAuthenticationError as e:
        logger.error(f"Authentication required: {e.recovery_hint}")
        logger.info(
            "Request access at: https://huggingface.co/datasets/Idavidrein/gpqa"
        )
        return {
            "success": False,
            "error": "Authentication required",
            "recovery": e.recovery_hint,
        }
    except Exception as e:
        logger.error(f"GPQA evaluation error: {e}")
        return {"success": False, "error": str(e)}


def evaluate_codeforces(
    models_config: List[Tuple[str, Any]], sample_size: int = 3
) -> Dict[str, Any]:
    """Evaluate models on Codeforces dataset.

    Args:
        models_config: List of (name, model) tuples to evaluate
        sample_size: Number of problems to sample

    Returns:
        Results dictionary
    """
    logger.info("Evaluating models on Codeforces dataset...")

    try:
        # Test with difficulty filtering
        cf_data = (
            DatasetBuilder()
            .from_registry("codeforces")
            .configure(difficulty_range=(800, 1200))
            .sample(sample_size)
            .build()
        )

        if len(cf_data) == 0:
            logger.error("Codeforces dataset loaded but contains no problems")
            return {"success": False, "error": "Empty dataset"}

        # Track results
        results = {"success": True, "model_results": {}, "problems": []}

        # Run evaluation (solution generation only, no execution)
        for i, problem in enumerate(problems := cf_data):
            logger.info(f"\nProblem {i+1}: {problem.query[:100]}...")
            logger.info(f"Difficulty: {problem.metadata.get('difficulty')}")

            problem_result = {
                "id": problem.metadata.get("id", f"problem_{i}"),
                "query": problem.query[:500] + "..."
                if len(problem.query) > 500
                else problem.query,
                "difficulty": problem.metadata.get("difficulty"),
                "tags": problem.metadata.get("tags", []),
                "model_responses": {},
            }

            # Generate solutions with each model (no evaluation)
            for name, model in models_config:
                if name not in results["model_results"]:
                    results["model_results"][name] = {
                        "solutions": 0,
                        "total": 0,
                        "time": 0,
                    }

                prompt = f"Solve this programming problem and provide a Python solution:\n\n{problem.query}\n\nReturn only the Python code solution inside ```python ``` code blocks."

                start_time = time.time()
                response = model(prompt)
                inference_time = time.time() - start_time

                # Extract code from response
                code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
                has_solution = bool(code_match)

                # Update results
                problem_result["model_responses"][name] = {
                    "response_preview": response[:200] + "..."
                    if len(response) > 200
                    else response,
                    "has_solution": has_solution,
                    "time": inference_time,
                }

                if has_solution:
                    results["model_results"][name]["solutions"] += 1
                results["model_results"][name]["total"] += 1
                results["model_results"][name]["time"] += inference_time

                logger.info(
                    f"{name}: {'✓' if has_solution else '✗'} solution generated [{inference_time:.2f}s]"
                )

            results["problems"].append(problem_result)

        # Calculate metrics
        for name in results["model_results"]:
            model_data = results["model_results"][name]
            model_data["solution_rate"] = (
                model_data["solutions"] / model_data["total"]
                if model_data["total"] > 0
                else 0
            )
            model_data["avg_time"] = (
                model_data["time"] / model_data["total"]
                if model_data["total"] > 0
                else 0
            )

        return results

    except Exception as e:
        logger.error(f"Codeforces evaluation error: {e}")
        return {"success": False, "error": str(e)}


def display_results(dataset_name: str, results: Dict[str, Any]) -> None:
    """Display evaluation results.

    Args:
        dataset_name: Name of the dataset
        results: Results dictionary
    """
    if not results.get("success", False):
        logger.error(
            f"{dataset_name} evaluation failed: {results.get('error', 'Unknown error')}"
        )
        if "recovery" in results:
            logger.info(f"Recovery: {results['recovery']}")
        return

    logger.info(f"\n----- {dataset_name} Results -----")
    logger.info(f"Problems evaluated: {len(results.get('problems', []))}")

    model_results = results.get("model_results", {})

    # Display model accuracy
    for name, data in model_results.items():
        if dataset_name.lower() == "codeforces":
            solution_rate = data.get("solution_rate", 0)
            solutions = data.get("solutions", 0)
            total = data.get("total", 0)
            avg_time = data.get("avg_time", 0)
            logger.info(
                f"{name}: {solutions}/{total} solutions generated ({solution_rate:.1%}) - {avg_time:.2f}s avg"
            )
        else:
            accuracy = data.get("accuracy", 0)
            correct = data.get("correct", 0)
            total = data.get("total", 0)
            avg_time = data.get("avg_time", 0)
            logger.info(
                f"{name}: {correct}/{total} correct ({accuracy:.1%}) - {avg_time:.2f}s avg"
            )


def visualize_comparison(all_results: Dict[str, Dict[str, Any]]) -> None:
    """Visualize performance comparison across datasets.

    Args:
        all_results: Dictionary with dataset results
    """
    if not HAS_VISUALIZATION:
        logger.warning("Skipping visualization - matplotlib not available")
        return

    # Extract data for visualization
    datasets = []
    model_names = set()

    # First, identify all models and valid datasets
    for dataset_name, result in all_results.items():
        if result.get("success", False):
            datasets.append(dataset_name)
            for model in result.get("model_results", {}):
                model_names.add(model)

    if not datasets or not model_names:
        logger.warning("No valid data for visualization")
        return

    # Convert to sorted lists for consistent ordering
    datasets = sorted(datasets)
    model_names = sorted(model_names)

    # Create accuracy matrix
    accuracy_data = []
    time_data = []

    for model in model_names:
        model_accuracy = []
        model_time = []
        for dataset in datasets:
            result = all_results.get(dataset, {})
            if result.get("success", False):
                model_result = result.get("model_results", {}).get(model, {})

                # Handle Codeforces differently (solution rate vs accuracy)
                if dataset.lower() == "codeforces":
                    accuracy = model_result.get("solution_rate", 0)
                else:
                    accuracy = model_result.get("accuracy", 0)

                model_accuracy.append(accuracy * 100)  # Convert to percentage
                model_time.append(model_result.get("avg_time", 0))
            else:
                model_accuracy.append(0)
                model_time.append(0)

        accuracy_data.append(model_accuracy)
        time_data.append(model_time)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Performance comparison
    bar_width = 0.8 / len(model_names)
    x = np.arange(len(datasets))

    for i, (model, accuracy) in enumerate(zip(model_names, accuracy_data)):
        offset = (i - len(model_names) / 2 + 0.5) * bar_width
        ax1.bar(x + offset, accuracy, bar_width, label=model)

    # Customize performance chart
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Model Performance by Dataset")
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.upper() for d in datasets])
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(model_names))
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.set_ylim(0, 105)  # Make room for labels

    # Add data labels
    for i, model_data in enumerate(accuracy_data):
        for j, v in enumerate(model_data):
            offset = (i - len(model_names) / 2 + 0.5) * bar_width
            ax1.text(
                j + offset, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=8
            )

    # Time comparison
    for i, (model, times) in enumerate(zip(model_names, time_data)):
        offset = (i - len(model_names) / 2 + 0.5) * bar_width
        ax2.bar(x + offset, times, bar_width, label=model)

    # Customize time chart
    ax2.set_ylabel("Average Time (seconds)")
    ax2.set_title("Model Inference Time by Dataset")
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.upper() for d in datasets])
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(model_names))
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Add data labels
    for i, model_data in enumerate(time_data):
        for j, v in enumerate(model_data):
            offset = (i - len(model_names) / 2 + 0.5) * bar_width
            ax2.text(
                j + offset, v + 0.1, f"{v:.1f}s", ha="center", va="bottom", fontsize=8
            )

    plt.tight_layout()
    plt.savefig("model_benchmark_results.png", dpi=300, bbox_inches="tight")
    logger.info("Visualization saved to model_benchmark_results.png")
    plt.show()


def main() -> None:
    """Run the model benchmark on specialized datasets."""
    parser = argparse.ArgumentParser(
        description="Benchmark models on specialized datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=["aime", "gpqa", "codeforces", "all"],
        default="all",
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--samples", type=int, default=5, help="Number of samples to evaluate"
    )
    args = parser.parse_args()

    # Configure models to evaluate
    models_config = [
        ("gpt-4o", models.openai.gpt4o()),
        ("claude-3-opus", models.anthropic.claude_3_opus()),
        ("claude-3-sonnet", models.anthropic.claude_3_sonnet()),
    ]

    results = {}

    # Run evaluations
    if args.dataset in ["aime", "all"]:
        results["AIME"] = evaluate_aime(models_config, args.samples)
        display_results("AIME", results["AIME"])

    if args.dataset in ["gpqa", "all"]:
        results["GPQA"] = evaluate_gpqa(models_config, args.samples)
        display_results("GPQA", results["GPQA"])

    if args.dataset in ["codeforces", "all"]:
        results["Codeforces"] = evaluate_codeforces(models_config, min(args.samples, 3))
        display_results("Codeforces", results["Codeforces"])

    # Generate visualization if multiple datasets were evaluated
    if len(results) > 1:
        visualize_comparison(results)

    # Check if any evaluations succeeded
    any_success = any(r.get("success", False) for r in results.values())
    if not any_success:
        logger.error("All evaluations failed")
        sys.exit(1)
    else:
        logger.info("\nBenchmark completed successfully")
        logger.info("See model_benchmark_results.png for visualization")


if __name__ == "__main__":
    main()
