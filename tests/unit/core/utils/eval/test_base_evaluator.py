"""Tests for base_evaluator module."""

from typing import Any, Dict, Optional, TypeVar
import sys
import os
import unittest
from unittest import mock

import pytest

# Print current path for debugging
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

try:
    from ember.core.utils.eval.base_evaluator import (
        EvaluationResult,
        IEvaluator,
        IStatefulEvaluator,
    )
except ImportError:
    from src.ember.core.utils.eval.base_evaluator import (
        EvaluationResult,
        IEvaluator,
        IStatefulEvaluator,
    )

# Type variables for generic testing
T_out = TypeVar("T_out")
T_ans = TypeVar("T_ans")


class TestEvaluationResult(unittest.TestCase):
    """Test suite for the EvaluationResult dataclass."""

    def test_initialization_with_required_fields(self) -> None:
        """Test initialization with only required fields."""
        # Arrange & Act
        result = EvaluationResult(is_correct=True, score=0.75)

        # Assert
        self.assertTrue(result.is_correct)
        self.assertEqual(0.75, result.score)
        self.assertIsNone(result.metadata)

    def test_initialization_with_metadata(self) -> None:
        """Test initialization with metadata."""
        # Arrange
        metadata: Dict[str, Any] = {"confidence": 0.9, "explanation": "Matches pattern"}

        # Act
        result = EvaluationResult(is_correct=False, score=0.0, metadata=metadata)

        # Assert
        self.assertFalse(result.is_correct)
        self.assertEqual(0.0, result.score)
        self.assertEqual(metadata, result.metadata)

    def test_metadata_is_optional(self) -> None:
        """Test that the metadata field is optional and defaults to None."""
        # Arrange & Act
        result1 = EvaluationResult(is_correct=True, score=1.0)
        result2 = EvaluationResult(is_correct=True, score=1.0, metadata=None)

        # Assert
        self.assertIsNone(result1.metadata)
        self.assertIsNone(result2.metadata)


class ConcreteEvaluator(IEvaluator[str, str]):
    """Concrete implementation of IEvaluator for testing."""

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: Any
    ) -> EvaluationResult:
        """Simple implementation that checks if the output equals the answer."""
        is_correct = system_output == correct_answer
        score = 1.0 if is_correct else 0.0
        return EvaluationResult(is_correct=is_correct, score=score)


class TestIEvaluator(unittest.TestCase):
    """Test suite for the IEvaluator interface."""

    def test_abstract_class_cannot_be_instantiated(self) -> None:
        """Test that IEvaluator cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            _ = IEvaluator()  # type: ignore

    def test_concrete_implementation(self) -> None:
        """Test a concrete implementation of IEvaluator."""
        # Arrange
        evaluator = ConcreteEvaluator()

        # Act
        result1 = evaluator.evaluate("hello", "hello")
        result2 = evaluator.evaluate("hello", "world")

        # Assert
        self.assertTrue(result1.is_correct)
        self.assertEqual(1.0, result1.score)

        self.assertFalse(result2.is_correct)
        self.assertEqual(0.0, result2.score)


class ConcreteStatefulEvaluator(IStatefulEvaluator[str, str]):
    """Concrete implementation of IStatefulEvaluator for testing."""

    def __init__(self) -> None:
        """Initialize with empty state."""
        self.total_correct = 0
        self.total_evaluated = 0

    def update(self, system_output: str, correct_answer: str, **kwargs: Any) -> None:
        """Update the state based on the evaluation."""
        self.total_evaluated += 1
        if system_output == correct_answer:
            self.total_correct += 1

    def compute(self) -> EvaluationResult:
        """Compute the final result based on accumulated state."""
        if self.total_evaluated == 0:
            return EvaluationResult(is_correct=False, score=0.0)

        accuracy = self.total_correct / self.total_evaluated
        return EvaluationResult(
            is_correct=accuracy == 1.0,
            score=accuracy,
            metadata={"total_evaluated": self.total_evaluated},
        )


class TestIStatefulEvaluator(unittest.TestCase):
    """Test suite for the IStatefulEvaluator interface."""

    def test_abstract_class_cannot_be_instantiated(self) -> None:
        """Test that IStatefulEvaluator cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            _ = IStatefulEvaluator()  # type: ignore

    def test_concrete_implementation_update_compute(self) -> None:
        """Test a concrete implementation using update and compute methods."""
        # Arrange
        evaluator = ConcreteStatefulEvaluator()

        # Act - update with a few samples
        evaluator.update("hello", "hello")
        evaluator.update("world", "world")
        evaluator.update("python", "java")
        result = evaluator.compute()

        # Assert
        self.assertFalse(result.is_correct)  # Not all are correct
        self.assertAlmostEqual(2 / 3, result.score)
        self.assertEqual({"total_evaluated": 3}, result.metadata)

    def test_concrete_implementation_evaluate(self) -> None:
        """Test the evaluate convenience method."""
        # Arrange
        evaluator = ConcreteStatefulEvaluator()

        # Act - evaluate a single sample
        result = evaluator.evaluate("hello", "hello")

        # Assert
        self.assertTrue(result.is_correct)
        self.assertEqual(1.0, result.score)
        self.assertEqual({"total_evaluated": 1}, result.metadata)

    def test_empty_state(self) -> None:
        """Test computing result with empty state."""
        # Arrange
        evaluator = ConcreteStatefulEvaluator()

        # Act
        result = evaluator.compute()

        # Assert
        self.assertFalse(result.is_correct)
        self.assertEqual(0.0, result.score)

    def test_evaluate_calls_update_and_compute(self) -> None:
        """Test that evaluate calls both update and compute."""
        # Arrange
        evaluator = ConcreteStatefulEvaluator()
        evaluator.update = mock.MagicMock()  # type: ignore
        evaluator.compute = mock.MagicMock(return_value=EvaluationResult(is_correct=True, score=1.0))  # type: ignore

        # Act
        result = evaluator.evaluate("output", "answer", custom_param=True)

        # Assert
        evaluator.update.assert_called_once_with("output", "answer", custom_param=True)
        evaluator.compute.assert_called_once()
        self.assertEqual(evaluator.compute.return_value, result)


if __name__ == "__main__":
    unittest.main()
