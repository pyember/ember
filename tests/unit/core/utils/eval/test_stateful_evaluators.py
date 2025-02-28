"""Tests for stateful_evaluators module."""

from typing import Any, Dict, List, Optional, Tuple, TypeVar
import sys
import os
import unittest
from unittest import mock

import pytest

# Print current path for debugging
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

try:
    from ember.core.utils.eval.base_evaluator import EvaluationResult, IEvaluator
except ImportError:
    from src.ember.core.utils.eval.base_evaluator import EvaluationResult, IEvaluator

try:
    from ember.core.utils.eval.stateful_evaluators import AggregatorEvaluator
except ImportError:
    from src.ember.core.utils.eval.stateful_evaluators import AggregatorEvaluator


class TestAggregatorEvaluator(unittest.TestCase):
    """Tests for the AggregatorEvaluator class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a mock evaluator
        self.mock_evaluator = mock.MagicMock(spec=IEvaluator)

        # Create an aggregator evaluator
        self.aggregator = AggregatorEvaluator(evaluator=self.mock_evaluator)

    def test_initialization(self) -> None:
        """Test that the aggregator is initialized with the provided evaluator."""
        # Assert
        self.assertEqual(self.mock_evaluator, self.aggregator.evaluator)
        self.assertEqual([], self.aggregator.results)

    def test_update_method(self) -> None:
        """Test the update method for accumulating results."""
        # Arrange
        self.mock_evaluator.evaluate.side_effect = [
            EvaluationResult(is_correct=True, score=1.0),
            EvaluationResult(is_correct=False, score=0.5),
        ]

        # Act
        self.aggregator.update("output1", "answer1", param1="value1")
        self.aggregator.update("output2", "answer2", param2="value2")

        # Assert
        self.assertEqual(2, len(self.aggregator.results))
        self.mock_evaluator.evaluate.assert_has_calls(
            [
                mock.call("output1", "answer1", param1="value1"),
                mock.call("output2", "answer2", param2="value2"),
            ]
        )
        self.assertTrue(self.aggregator.results[0].is_correct)
        self.assertFalse(self.aggregator.results[1].is_correct)

    def test_compute_method_empty(self) -> None:
        """Test computing result with no accumulated data."""
        # Act
        result = self.aggregator.compute()

        # Assert
        self.assertFalse(result.is_correct)
        self.assertEqual(0.0, result.score)
        self.assertEqual({"accuracy": 0.0, "total_samples": 0}, result.metadata)

    def test_compute_method_with_data(self) -> None:
        """Test computing result with accumulated data."""
        # Arrange
        self.aggregator.results = [
            EvaluationResult(is_correct=True, score=1.0),
            EvaluationResult(is_correct=True, score=0.8),
            EvaluationResult(is_correct=False, score=0.4),
        ]

        # Act
        result = self.aggregator.compute()

        # Assert
        self.assertFalse(result.is_correct)  # Not all results are correct
        self.assertAlmostEqual((1.0 + 0.8 + 0.4) / 3, result.score)
        self.assertEqual({"accuracy": 2 / 3, "total_samples": 3}, result.metadata)

    def test_compute_method_all_correct(self) -> None:
        """Test computing result when all accumulated results are correct."""
        # Arrange
        self.aggregator.results = [
            EvaluationResult(is_correct=True, score=1.0),
            EvaluationResult(is_correct=True, score=0.8),
            EvaluationResult(is_correct=True, score=0.9),
        ]

        # Act
        result = self.aggregator.compute()

        # Assert
        self.assertTrue(result.is_correct)  # All results are correct
        self.assertAlmostEqual((1.0 + 0.8 + 0.9) / 3, result.score)
        self.assertEqual({"accuracy": 1.0, "total_samples": 3}, result.metadata)

    def test_evaluate_method(self) -> None:
        """Test the evaluate convenience method."""
        # Arrange
        self.mock_evaluator.evaluate.return_value = EvaluationResult(
            is_correct=True, score=1.0
        )

        # Act
        with mock.patch.object(self.aggregator, "update") as mock_update:
            with mock.patch.object(self.aggregator, "compute") as mock_compute:
                mock_compute.return_value = EvaluationResult(
                    is_correct=True, score=0.9, metadata={"total_samples": 1}
                )

                result = self.aggregator.evaluate("output", "answer", param="value")

        # Assert
        mock_update.assert_called_once_with("output", "answer", param="value")
        mock_compute.assert_called_once()
        self.assertEqual(mock_compute.return_value, result)

    def test_end_to_end(self) -> None:
        """Test the evaluator from end to end without mocking internal methods."""
        # Arrange
        self.mock_evaluator.evaluate.side_effect = [
            EvaluationResult(is_correct=True, score=1.0),
            EvaluationResult(is_correct=False, score=0.0),
            EvaluationResult(is_correct=True, score=0.8),
        ]

        # Act - Call evaluate multiple times
        result1 = self.aggregator.evaluate("output1", "answer1")
        result2 = self.aggregator.evaluate("output2", "answer2")
        result3 = self.aggregator.evaluate("output3", "answer3")

        # Assert
        self.assertEqual(3, len(self.aggregator.results))

        # First evaluation should return a perfect score (only one sample)
        self.assertTrue(result1.is_correct)
        self.assertEqual(1.0, result1.score)
        self.assertEqual({"accuracy": 1.0, "total_samples": 1}, result1.metadata)

        # Second evaluation should show 50% accuracy
        self.assertFalse(result2.is_correct)
        self.assertEqual(0.5, result2.score)
        self.assertEqual({"accuracy": 0.5, "total_samples": 2}, result2.metadata)

        # Third evaluation should show 2/3 accuracy
        self.assertFalse(result3.is_correct)
        self.assertAlmostEqual(0.6, result3.score)
        self.assertEqual({"accuracy": 2 / 3, "total_samples": 3}, result3.metadata)

    def test_with_real_evaluator(self) -> None:
        """Test with a real evaluator rather than a mock."""

        # Arrange
        class SimpleEvaluator(IEvaluator[str, str]):
            def evaluate(
                self, system_output: str, correct_answer: str, **kwargs: Any
            ) -> EvaluationResult:
                is_correct = system_output == correct_answer
                score = 1.0 if is_correct else 0.0
                return EvaluationResult(is_correct=is_correct, score=score)

        aggregator = AggregatorEvaluator(evaluator=SimpleEvaluator())

        # Act
        aggregator.update("hello", "hello")
        aggregator.update("world", "world")
        aggregator.update("python", "java")
        result = aggregator.compute()

        # Assert
        self.assertFalse(result.is_correct)
        self.assertAlmostEqual(2 / 3, result.score)
        self.assertEqual({"accuracy": 2 / 3, "total_samples": 3}, result.metadata)


if __name__ == "__main__":
    unittest.main()
