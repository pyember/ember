"""Integration tests for the eval module components working together."""

import unittest
import sys
import os
from typing import Any, Dict, List, Tuple, Type

# Print current path for debugging
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

# Try both import paths
try:
    from ember.core.utils.eval.base_evaluator import EvaluationResult, IEvaluator
except ImportError:
    print("Trying alternative import path...")
    try:
        from src.ember.core.utils.eval.base_evaluator import (
            EvaluationResult,
            IEvaluator,
        )
    except ImportError as e:
        print(f"Import error: {e}")
        raise

import pytest

# Continue with the rest of the imports
try:
    from ember.core.utils.eval.evaluators import (
        ComposedEvaluator,
        ExactMatchEvaluator,
        NumericToleranceEvaluator,
        PartialRegexEvaluator,
    )
except ImportError:
    from src.ember.core.utils.eval.evaluators import (
        ComposedEvaluator,
        ExactMatchEvaluator,
        NumericToleranceEvaluator,
        PartialRegexEvaluator,
    )

try:
    from ember.core.utils.eval.extractors import RegexExtractor
except ImportError:
    from src.ember.core.utils.eval.extractors import RegexExtractor

try:
    from ember.core.utils.eval.pipeline import (
        PipelineEvaluator,
        evaluate_batch_with_summary,
    )
except ImportError:
    from src.ember.core.utils.eval.pipeline import (
        PipelineEvaluator,
        evaluate_batch_with_summary,
    )

try:
    from ember.core.utils.eval.registry import EvaluatorRegistry
except ImportError:
    from src.ember.core.utils.eval.registry import EvaluatorRegistry

try:
    from ember.core.utils.eval.stateful_evaluators import AggregatorEvaluator
except ImportError:
    from src.ember.core.utils.eval.stateful_evaluators import AggregatorEvaluator


class TestComponentIntegration(unittest.TestCase):
    """Tests that verify different components work correctly together."""

    def test_pipeline_with_regex_extract(self) -> None:
        """Test a pipeline that extracts values using regex and performs numeric evaluation."""

        # Arrange - Set up a pipeline to extract numbers from text and compare with tolerance
        def extract_number(text: str) -> float:
            """Extract a numeric value from text using regex."""
            pattern = r"value is (\d+\.?\d*)"
            match = RegexExtractor(pattern=pattern).extract(text)
            return float(match) if match else 0.0

        pipeline = PipelineEvaluator(
            transforms=[extract_number],
            evaluator=NumericToleranceEvaluator(tolerance=0.1),
        )

        # Act
        result1 = pipeline.evaluate("The value is 42.5", 42.5)
        result2 = pipeline.evaluate("The value is 42.55", 42.5)
        result3 = pipeline.evaluate("The value is 42.7", 42.5)

        # Assert
        self.assertTrue(result1.is_correct)
        self.assertEqual(1.0, result1.score)

        self.assertTrue(result2.is_correct)
        self.assertAlmostEqual(0.999, result2.score, places=2)

        self.assertFalse(result3.is_correct)
        self.assertAlmostEqual(0.995, result3.score, places=3)

    def test_composed_evaluator_and_registry(self) -> None:
        """Test using the registry to create composed evaluators."""
        # Arrange - Create a registry with factory methods
        registry = EvaluatorRegistry()

        def create_exact_match() -> IEvaluator[str, str]:
            return ExactMatchEvaluator()

        def create_regex_evaluator(pattern: str) -> IEvaluator[str, str]:
            return PartialRegexEvaluator(pattern=pattern)

        def create_numeric_evaluator(
            tolerance: float = 0.01,
        ) -> IEvaluator[float, float]:
            return NumericToleranceEvaluator(tolerance=tolerance)

        # Register the factories
        registry.register("exact_match", create_exact_match)
        registry.register("regex", create_regex_evaluator)
        registry.register("numeric", create_numeric_evaluator)

        # Act - Create evaluators from the registry
        exact_evaluator = registry.create("exact_match")
        regex_evaluator = registry.create("regex", pattern=r"answer is (\w+)")
        numeric_evaluator = registry.create("numeric", tolerance=0.05)

        # Evaluate some inputs
        exact_result = exact_evaluator.evaluate("Hello", "hello")
        regex_result = regex_evaluator.evaluate("The answer is Paris", "Paris")
        numeric_result = numeric_evaluator.evaluate(10.03, 10.0)

        # Assert
        self.assertTrue(exact_result.is_correct)
        self.assertTrue(regex_result.is_correct)
        self.assertTrue(numeric_result.is_correct)

    def test_stateful_with_pipeline(self) -> None:
        """Test using a stateful evaluator with a pipeline evaluator."""

        # Arrange - Create a pipeline evaluator and wrap in stateful evaluator
        def extract_number(text: str) -> float:
            """Extract a numeric value from text using regex."""
            pattern = r"value is (\d+\.?\d*)"
            match = RegexExtractor(pattern=pattern).extract(text)
            return float(match) if match else 0.0

        pipeline = PipelineEvaluator(
            transforms=[extract_number],
            evaluator=NumericToleranceEvaluator(tolerance=0.1),
        )

        stateful = AggregatorEvaluator(evaluator=pipeline)

        # Act - Accumulate multiple evaluations
        stateful.update("The value is 42.5", 42.5)
        stateful.update("The value is 42.55", 42.5)
        stateful.update("The value is 42.7", 42.5)
        stateful.update("value is invalid", 42.5)  # This should fail extraction

        result = stateful.compute()

        # Assert
        self.assertFalse(result.is_correct)  # Not all evaluations were correct
        self.assertLess(result.score, 1.0)
        self.assertEqual(0.5, result.metadata["accuracy"])  # 2 out of 4 correct
        self.assertEqual(4, result.metadata["total_samples"])

    def test_batch_evaluation_integration(self) -> None:
        """Test batch evaluation with various evaluators."""
        # Arrange - Create different evaluator types
        evaluators: Dict[str, IEvaluator[Any, Any]] = {
            "exact": ExactMatchEvaluator(),
            "regex": PartialRegexEvaluator(pattern=r"Capital of \w+ is (\w+)"),
            "numeric": NumericToleranceEvaluator(tolerance=0.1),
        }

        # Prepare test data for each evaluator type
        test_cases: Dict[str, Tuple[List[Any], List[Any]]] = {
            "exact": (["Paris", "London", "Berlin"], ["Paris", "London", "Rome"]),
            "regex": (
                [
                    "Capital of France is Paris",
                    "Capital of UK is London",
                    "Capital of Italy is Rome",
                ],
                ["Paris", "London", "Rome"],
            ),
            "numeric": ([10.0, 20.05, 30.2], [10.0, 20.0, 30.0]),
        }

        # Act - Run batch evaluations for each evaluator type
        results: Dict[str, Dict[str, float]] = {}

        for eval_type, evaluator in evaluators.items():
            system_outputs, correct_answers = test_cases[eval_type]
            summary = evaluate_batch_with_summary(
                evaluator=evaluator,
                system_outputs=system_outputs,
                correct_answers=correct_answers,
            )
            results[eval_type] = {
                "mean_score": summary.mean_score,
                "accuracy": summary.accuracy,
            }

        # Assert
        self.assertEqual(2 / 3, results["exact"]["accuracy"])
        self.assertEqual(1.0, results["regex"]["accuracy"])
        self.assertEqual(2 / 3, results["numeric"]["accuracy"])


class TestRealWorldScenarios(unittest.TestCase):
    """Tests that simulate real-world evaluation scenarios."""

    def test_qa_evaluation_scenario(self) -> None:
        """Test a question-answering evaluation scenario."""
        # Arrange - Create a QA dataset with questions, responses, and answers
        qa_data = [
            {
                "question": "What is the capital of France?",
                "response": "The capital of France is Paris.",
                "answer": "Paris",
            },
            {
                "question": "Who wrote 'Pride and Prejudice'?",
                "response": "Jane Austen wrote 'Pride and Prejudice'.",
                "answer": "Jane Austen",
            },
            {
                "question": "What is 7 * 8?",
                "response": "7 * 8 = 56",
                "answer": "56",
            },
            {
                "question": "What is the speed of light?",
                "response": "The speed of light is approximately 299,792,458 meters per second.",
                "answer": "299792458",
            },
            {
                "question": "When was the Declaration of Independence signed?",
                "response": "The Declaration of Independence was primarily signed on August 2, 1776.",
                "answer": "1776",
            },
        ]

        # Create patterns to extract answers from responses
        answer_patterns = {
            "capital": r"capital of \w+ is (\w+)",
            "author": r"(\w+\s+\w+) wrote",
            "math": r"=\s*(\d+)",
            "number": r"approximately ([\d,]+)",
            "year": r"(\d{4})",
        }

        # Create evaluators for different question types
        evaluators: Dict[str, IEvaluator[str, str]] = {
            "capital": PartialRegexEvaluator(pattern=answer_patterns["capital"]),
            "author": PartialRegexEvaluator(pattern=answer_patterns["author"]),
            "math": PartialRegexEvaluator(pattern=answer_patterns["math"]),
            "number": ComposedEvaluator(
                extractor=RegexExtractor(pattern=answer_patterns["number"]),
                base_evaluator=ExactMatchEvaluator(
                    compare_fn=lambda x, y: x.replace(",", "") == y
                ),
            ),
            "year": PartialRegexEvaluator(pattern=answer_patterns["year"]),
        }

        # Map questions to evaluator types
        question_types = ["capital", "author", "math", "number", "year"]

        # Act - Evaluate each response with the appropriate evaluator
        results = []
        for i, item in enumerate(qa_data):
            question_type = question_types[i]
            evaluator = evaluators[question_type]
            result = evaluator.evaluate(item["response"], item["answer"])
            results.append(result)

        # Create a summary
        correct_count = sum(1 for r in results if r.is_correct)
        accuracy = correct_count / len(results)

        # Assert
        self.assertEqual(5, len(results))
        self.assertTrue(results[0].is_correct)  # capital
        self.assertTrue(results[1].is_correct)  # author
        self.assertTrue(results[2].is_correct)  # math
        self.assertTrue(results[3].is_correct)  # number
        self.assertTrue(results[4].is_correct)  # year
        self.assertEqual(1.0, accuracy)

    def test_registry_with_custom_evaluators(self) -> None:
        """Test creating and using custom evaluators through the registry."""

        # Arrange - Create custom evaluators for specific tasks
        class MultipleChoiceEvaluator(IEvaluator[str, str]):
            """Evaluator for multiple choice answers (A, B, C, D)."""

            def evaluate(
                self, system_output: str, correct_answer: str, **kwargs: Any
            ) -> EvaluationResult:
                # Extract first letter that matches A, B, C, or D
                import re

                # Look for choices in the format of "A) ", "B.", "answer is B", etc.
                match = re.search(
                    r"(?:^|\s+|answer\s+is\s+)([ABCD])(?:$|\s+|\)|\.)",
                    system_output.upper(),
                )
                extracted = match.group(1) if match else ""
                is_correct = extracted == correct_answer.upper()
                return EvaluationResult(
                    is_correct=is_correct, score=1.0 if is_correct else 0.0
                )

        class SemanticSimilarityEvaluator(IEvaluator[str, str]):
            """Simulated evaluator for semantic similarity."""

            def __init__(self, threshold: float = 0.7) -> None:
                self.threshold = threshold

            def evaluate(
                self, system_output: str, correct_answer: str, **kwargs: Any
            ) -> EvaluationResult:
                # For this test, simulate similarity based on word overlap
                output_words = set(system_output.lower().split())
                answer_words = set(correct_answer.lower().split())

                if not output_words or not answer_words:
                    return EvaluationResult(is_correct=False, score=0.0)

                common_words = output_words.intersection(answer_words)
                similarity = len(common_words) / max(
                    len(output_words), len(answer_words)
                )

                is_correct = similarity >= self.threshold
                return EvaluationResult(
                    is_correct=is_correct,
                    score=similarity,
                    metadata={"similarity": similarity},
                )

        # Create a registry
        registry = EvaluatorRegistry()

        # Register custom evaluator factories
        def create_multiple_choice() -> IEvaluator[str, str]:
            return MultipleChoiceEvaluator()

        def create_semantic_similarity(threshold: float = 0.7) -> IEvaluator[str, str]:
            return SemanticSimilarityEvaluator(threshold=threshold)

        def create_exact_match() -> IEvaluator[str, str]:
            return ExactMatchEvaluator()

        registry.register("multiple_choice", create_multiple_choice)
        registry.register("semantic", create_semantic_similarity)
        registry.register("exact", create_exact_match)

        # Act - Create and use evaluators for different question types
        mc_evaluator = registry.create("multiple_choice")
        semantic_evaluator = registry.create("semantic", threshold=0.5)
        exact_evaluator = registry.create("exact")

        mc_result = mc_evaluator.evaluate("I think the answer is B.", "B")
        semantic_result = semantic_evaluator.evaluate(
            "The speed of light is very fast", "Light speed is extremely fast"
        )
        exact_result = exact_evaluator.evaluate("Paris", "Paris")

        # Assert
        self.assertTrue(mc_result.is_correct)
        self.assertTrue(semantic_result.is_correct)
        self.assertTrue(exact_result.is_correct)

        # Test accuracy with batch evaluation
        test_data = [
            {"output": "The answer is A", "answer": "A", "type": "multiple_choice"},
            {
                "output": "Birds can fly in the sky",
                "answer": "Birds fly in sky",
                "type": "semantic",
            },
            {"output": "42", "answer": "42", "type": "exact"},
            {
                "output": "I believe C is correct",
                "answer": "B",
                "type": "multiple_choice",
            },
            {
                "output": "The Earth orbits the sun",
                "answer": "The moon orbits the Earth",
                "type": "semantic",
            },
        ]

        results = []
        for item in test_data:
            evaluator = registry.create(item["type"])
            result = evaluator.evaluate(item["output"], item["answer"])
            results.append(result)

        accuracy = sum(1 for r in results if r.is_correct) / len(results)
        self.assertEqual(3 / 5, accuracy)


if __name__ == "__main__":
    unittest.main()
