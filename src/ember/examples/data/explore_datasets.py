"""Tool for exploring Ember's specialized datasets (AIME, GPQA, Codeforces).

This utility helps users explore sample problems from each specialized dataset,
understand their structure, and test their integration with the Ember framework.

Usage:
    python -m ember.examples.data.explore_datasets --dataset aime --count 3
"""

import argparse
import logging
import sys
from typing import Dict, List, Any, Optional

from ember.api import datasets, DatasetBuilder

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def explore_aime(count: int = 3) -> None:
    """Explore AIME dataset structure.

    Args:
        count: Number of examples to show
    """
    try:
        logger.info("=== AIME Dataset Exploration ===")

        # Load AIME dataset
        aime_data = datasets("aime")
        logger.info(f"Successfully loaded {len(aime_data)} AIME problems")

        if not aime_data:
            logger.error("Dataset loaded but contains no problems")
            return

        # Sample problems
        sample_size = min(count, len(aime_data))
        logger.info(f"\nShowing {sample_size} sample AIME problems:")

        for i, problem in enumerate(aime_data[:sample_size]):
            problem_id = problem.metadata.get("problem_id", f"Problem {i+1}")
            logger.info(f"\n--- Problem {i+1}: {problem_id} ---")

            # Print problem
            logger.info(f"Query: {problem.query}\n")

            # Print answer
            answer = problem.metadata.get("correct_answer", "Unknown")
            logger.info(f"Answer: {answer}")

            # Print metadata
            logger.info("Metadata:")
            year = problem.metadata.get("year", "Unknown")
            contest = problem.metadata.get("contest", "Unknown")
            logger.info(f"  Year: {year}")
            logger.info(f"  Contest: {contest}")

            if i < sample_size - 1:
                logger.info("\n" + "-" * 40)

        # Show filtering example
        logger.info("\n=== Filtering Example ===")

        # Filter to AIME I problems
        from ember.core.utils.data.base.config import BaseDatasetConfig

        class AIMEConfig(BaseDatasetConfig):
            contest: str = None

        # Load dataset first, then filter manually
        aime_data = datasets("aime")

        # Filter to AIME I problems using config method
        aime_i = DatasetBuilder().from_registry("aime").config(contest="I").build()
        logger.info(f"AIME I problems: {len(aime_i)}")

        # Filter to AIME II problems
        aime_ii = DatasetBuilder().from_registry("aime").config(contest="II").build()
        logger.info(f"AIME II problems: {len(aime_ii)}")

    except Exception as e:
        logger.error(f"Error exploring AIME dataset: {e}")


def explore_gpqa(count: int = 3) -> None:
    """Explore GPQA dataset structure.

    Args:
        count: Number of examples to show
    """
    try:
        logger.info("=== GPQA Dataset Exploration ===")

        # Load GPQA dataset
        gpqa_data = datasets("gpqa")
        logger.info(f"Successfully loaded {len(gpqa_data)} GPQA questions")

        if not gpqa_data:
            logger.error("Dataset loaded but contains no questions")
            return

        # Sample questions
        sample_size = min(count, len(gpqa_data))
        logger.info(f"\nShowing {sample_size} sample GPQA questions:")

        for i, question in enumerate(gpqa_data[:sample_size]):
            question_id = question.metadata.get("id", f"Question {i+1}")
            logger.info(f"\n--- Question {i+1}: {question_id} ---")

            # Print question
            logger.info(f"Query: {question.query[:500]}...\n")

            # Print choices
            logger.info("Choices:")
            for key, choice in question.choices.items():
                logger.info(
                    f"  {key}: {choice[:100]}..."
                    if len(choice) > 100
                    else f"  {key}: {choice}"
                )

            # Print answer
            answer = question.metadata.get("correct_answer", "Unknown")
            logger.info(f"\nCorrect Answer: {answer}")

            # Print metadata
            logger.info("Metadata:")
            subject = question.metadata.get("subject", "Unknown")
            difficulty = question.metadata.get("difficulty", "Unknown")
            logger.info(f"  Subject: {subject}")
            logger.info(f"  Difficulty: {difficulty}")

            if i < sample_size - 1:
                logger.info("\n" + "-" * 40)

        # Show filtering example
        logger.info("\n=== Filtering Example ===")

        # Filter by subject
        physics_questions = (
            DatasetBuilder()
            .from_registry("gpqa")
            .filter(lambda item: "physics" in item.metadata.get("subject", "").lower())
            .build()
        )
        logger.info(f"Physics questions: {len(physics_questions)}")

        chemistry_questions = (
            DatasetBuilder()
            .from_registry("gpqa")
            .filter(
                lambda item: "chemistry" in item.metadata.get("subject", "").lower()
            )
            .build()
        )
        logger.info(f"Chemistry questions: {len(chemistry_questions)}")

    except Exception as e:
        logger.error(f"Error exploring GPQA dataset: {e}")
        if "gated dataset" in str(e) or "authentication" in str(e):
            logger.error("\nAuthentication required for GPQA dataset:")
            logger.error("1. Run: huggingface-cli login")
            logger.error(
                "2. Request access at: https://huggingface.co/datasets/Idavidrein/gpqa"
            )


def explore_codeforces(count: int = 3) -> None:
    """Explore Codeforces dataset structure.

    Args:
        count: Number of examples to show
    """
    try:
        logger.info("=== Codeforces Dataset Exploration ===")

        # Load Codeforces dataset with filtering
        cf_data = (
            DatasetBuilder()
            .from_registry("codeforces")
            .configure(difficulty_range=(800, 1200))  # Beginner-friendly
            .build()
        )
        logger.info(f"Successfully loaded {len(cf_data)} Codeforces problems")

        if not cf_data:
            logger.error("Dataset loaded but contains no problems")
            return

        # Sample problems
        sample_size = min(count, len(cf_data))
        logger.info(f"\nShowing {sample_size} sample Codeforces problems:")

        for i, problem in enumerate(cf_data[:sample_size]):
            problem_id = problem.metadata.get("id", f"Problem {i+1}")
            name = problem.metadata.get("name", "Unnamed problem")
            logger.info(f"\n--- Problem {i+1}: {problem_id} - {name} ---")

            # Print problem (truncated)
            query_preview = (
                problem.query[:500] + "..."
                if len(problem.query) > 500
                else problem.query
            )
            logger.info(f"Query: {query_preview}\n")

            # Print metadata
            logger.info("Metadata:")
            difficulty = problem.metadata.get("difficulty", "Unknown")
            tags = problem.metadata.get("tags", [])
            logger.info(f"  Difficulty: {difficulty}")
            logger.info(f"  Tags: {', '.join(tags)}")

            # Print test cases
            test_cases = problem.metadata.get("test_cases", [])
            if test_cases:
                logger.info(f"\nTest Cases ({len(test_cases)}):")
                for j, test in enumerate(
                    test_cases[:2]
                ):  # Show only first 2 test cases
                    logger.info(f"  Test {j+1}:")
                    logger.info(f"    Input: {test.get('input', 'N/A')}")
                    logger.info(f"    Expected Output: {test.get('output', 'N/A')}")

                if len(test_cases) > 2:
                    logger.info(f"  ... and {len(test_cases) - 2} more test cases")

            if i < sample_size - 1:
                logger.info("\n" + "-" * 40)

        # Show filtering example
        logger.info("\n=== Filtering Examples ===")

        # Filter by difficulty using config
        easy_problems = (
            DatasetBuilder()
            .from_registry("codeforces")
            .config(difficulty_range=(800, 1200))
            .build()
        )
        logger.info(f"Easy problems (800-1200): {len(easy_problems)}")

        # Medium difficulty problems
        medium_problems = (
            DatasetBuilder()
            .from_registry("codeforces")
            .config(difficulty_range=(1300, 1700))
            .build()
        )
        logger.info(f"Medium problems (1300-1700): {len(medium_problems)}")

        # Filter by tag
        dp_problems = (
            DatasetBuilder().from_registry("codeforces").config(tags=["dp"]).build()
        )
        logger.info(f"Dynamic Programming problems: {len(dp_problems)}")

    except Exception as e:
        logger.error(f"Error exploring Codeforces dataset: {e}")


def main() -> None:
    """Explore specialized datasets in Ember."""
    parser = argparse.ArgumentParser(description="Explore Ember's specialized datasets")
    parser.add_argument(
        "--dataset",
        choices=["aime", "gpqa", "codeforces", "all"],
        default="all",
        help="Dataset to explore",
    )
    parser.add_argument(
        "--count", type=int, default=3, help="Number of examples to show"
    )
    args = parser.parse_args()

    # Explore requested datasets
    if args.dataset == "all" or args.dataset == "aime":
        explore_aime(args.count)
        if args.dataset != "all":
            return

    if args.dataset == "all" or args.dataset == "gpqa":
        if args.dataset == "all":
            logger.info("\n\n" + "=" * 50 + "\n")
        explore_gpqa(args.count)
        if args.dataset != "all":
            return

    if args.dataset == "all" or args.dataset == "codeforces":
        if args.dataset == "all":
            logger.info("\n\n" + "=" * 50 + "\n")
        explore_codeforces(args.count)


if __name__ == "__main__":
    main()
