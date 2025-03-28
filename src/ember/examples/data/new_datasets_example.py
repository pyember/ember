"""Example for using specialized datasets (AIME, GPQA, Codeforces).

Run with: uvx python -m ember.examples.data.new_datasets_example [--skip-model-calls]
"""

import argparse
import logging
import sys
from typing import Dict, List, Optional

from ember.api import datasets, DatasetBuilder, models
from ember.core.exceptions import GatedDatasetAuthenticationError
from ember.core.utils.eval.numeric_answer import AIMEAnswerEvaluator
from ember.core.utils.eval.evaluators import MultipleChoiceEvaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_aime(skip_model_calls: bool = False) -> bool:
    """Test AIME dataset loading and evaluation.

    Args:
        skip_model_calls: If True, skip LLM evaluation

    Returns:
        Success status
    """
    logger.info("Testing AIME dataset...")

    try:
        aime_data = datasets("aime")
        logger.info(f"✓ Loaded {len(aime_data)} AIME problems")

        if len(aime_data) == 0:
            logger.error("Dataset loaded but contains no problems")
            return False

        # Sample problem
        problem = aime_data[0]
        logger.info(f"Sample problem: {problem.query[:100]}...")
        logger.info(f"Expected answer: {problem.metadata['correct_answer']}")

        # Test filtering
        aime_i = DatasetBuilder().from_registry("aime").config(contest="I").build()
        logger.info(f"✓ Filtered to {len(aime_i)} AIME-I problems")

        if not skip_model_calls:
            # Test evaluation
            logger.info("Testing model evaluation...")
            model = models.openai.gpt4o()
            response = model(problem.query)

            evaluator = AIMEAnswerEvaluator()
            result = evaluator.evaluate(response, problem.metadata["correct_answer"])
            logger.info(
                f"Model evaluation: {'correct' if result.is_correct else 'incorrect'}"
            )

        return True

    except Exception as e:
        logger.error(f"AIME dataset error: {e}")
        return False


def test_gpqa(skip_model_calls: bool = False) -> bool:
    """Test GPQA dataset loading and evaluation.

    Args:
        skip_model_calls: If True, skip LLM evaluation

    Returns:
        Success status
    """
    logger.info("Testing GPQA dataset...")

    try:
        gpqa_data = datasets("gpqa")
        logger.info(f"✓ Loaded {len(gpqa_data)} GPQA problems")

        if len(gpqa_data) == 0:
            logger.error("Dataset loaded but contains no questions")
            return False

        # Sample problem
        problem = gpqa_data[0]
        logger.info(f"Sample question: {problem.query[:100]}...")
        logger.info(f"Choices: {list(problem.choices.keys())}")
        logger.info(f"Expected answer: {problem.metadata['correct_answer']}")

        # Format prompt for evaluation
        prompt = problem.query + "\n\n"
        for key, choice in problem.choices.items():
            prompt += f"{key}. {choice}\n"

        if not skip_model_calls:
            # Test evaluation
            logger.info("Testing model evaluation...")
            model = models.openai.gpt4o()
            response = model(prompt)

            evaluator = MultipleChoiceEvaluator()
            result = evaluator.evaluate(response, problem.metadata["correct_answer"])
            logger.info(
                f"Model evaluation: {'correct' if result.is_correct else 'incorrect'}"
            )

        return True

    except GatedDatasetAuthenticationError as e:
        logger.error(f"Authentication required: {e.recovery_hint}")
        logger.info(
            "Request access at: https://huggingface.co/datasets/Idavidrein/gpqa"
        )
        return False
    except Exception as e:
        logger.error(f"GPQA dataset error: {e}")
        return False


def test_codeforces(skip_model_calls: bool = False) -> bool:
    """Test Codeforces dataset loading and evaluation.

    Args:
        skip_model_calls: If True, skip LLM evaluation

    Returns:
        Success status
    """
    logger.info("Testing Codeforces dataset...")

    try:
        # Test with difficulty filtering
        cf_data = (
            DatasetBuilder()
            .from_registry("codeforces")
            .config(difficulty_range=(800, 1200))
            .sample(5)
            .build()
        )
        logger.info(f"✓ Loaded {len(cf_data)} Codeforces problems")

        if len(cf_data) == 0:
            logger.error("Dataset loaded but contains no problems")
            return False

        # Sample problem
        problem = cf_data[0]
        logger.info(f"Sample problem: {problem.query[:100]}...")
        logger.info(f"Difficulty: {problem.metadata.get('difficulty')}")
        logger.info(f"Tags: {problem.metadata.get('tags', [])}")

        # Skip model evaluation for code problems to avoid long execution
        if not skip_model_calls:
            logger.info("Skipping code evaluation to avoid long execution")

        return True

    except Exception as e:
        logger.error(f"Codeforces dataset error: {e}")
        return False


def summary(results: Dict[str, bool]) -> None:
    """Print summary of test results.

    Args:
        results: Dictionary of test results
    """
    logger.info("\n--- Dataset Test Summary ---")

    all_passed = True
    for dataset, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{dataset}: {status}")
        all_passed = all_passed and success

    if all_passed:
        logger.info("\nAll datasets loaded successfully!")
    else:
        logger.warning("\nSome datasets failed to load. See errors above.")

    # Show authentication hint if GPQA failed
    if not results.get("gpqa", True):
        logger.info("\nFor GPQA authentication:")
        logger.info("1. Run: huggingface-cli login")
        logger.info(
            "2. Request access: https://huggingface.co/datasets/Idavidrein/gpqa"
        )


def main() -> None:
    """Run the dataset test example."""
    parser = argparse.ArgumentParser(description="Test specialized datasets")
    parser.add_argument(
        "--skip-model-calls",
        action="store_true",
        help="Skip any calls to language models",
    )
    args = parser.parse_args()

    results = {}

    results["aime"] = test_aime(args.skip_model_calls)
    results["gpqa"] = test_gpqa(args.skip_model_calls)
    results["codeforces"] = test_codeforces(args.skip_model_calls)

    summary(results)

    # Return non-zero exit code if any test failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
