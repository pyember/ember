"""Tests for registry module."""

from typing import Any
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
    from ember.core.utils.eval.base_evaluator import EvaluationResult, IEvaluator

try:
    from ember.core.utils.eval.registry import EvaluatorRegistry
except ImportError:
    from ember.core.utils.eval.registry import EvaluatorRegistry


class TestEvaluatorRegistry(unittest.TestCase):
    """Tests for the EvaluatorRegistry class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.registry = EvaluatorRegistry()

        # Create some mock evaluator factories
        self.mock_factory1 = mock.MagicMock(
            return_value=mock.MagicMock(spec=IEvaluator)
        )
        self.mock_factory2 = mock.MagicMock(
            return_value=mock.MagicMock(spec=IEvaluator)
        )

    def test_initialization(self) -> None:
        """Test that the registry is initialized with an empty registry dict."""
        # Arrange & Act
        registry = EvaluatorRegistry()

        # Assert
        self.assertEqual({}, registry._registry)

    def test_register_factory(self) -> None:
        """Test registering an evaluator factory."""
        # Act
        self.registry.register("factory1", self.mock_factory1)

        # Assert
        self.assertIn("factory1", self.registry._registry)
        self.assertEqual(self.mock_factory1, self.registry._registry["factory1"])

    def test_register_multiple_factories(self) -> None:
        """Test registering multiple evaluator factories."""
        # Act
        self.registry.register("factory1", self.mock_factory1)
        self.registry.register("factory2", self.mock_factory2)

        # Assert
        self.assertEqual(2, len(self.registry._registry))
        self.assertEqual(self.mock_factory1, self.registry._registry["factory1"])
        self.assertEqual(self.mock_factory2, self.registry._registry["factory2"])

    def test_overwrite_factory(self) -> None:
        """Test overwriting an existing factory registration."""
        # Arrange
        self.registry.register("factory1", self.mock_factory1)

        # Act
        self.registry.register("factory1", self.mock_factory2)

        # Assert
        self.assertEqual(1, len(self.registry._registry))
        self.assertEqual(self.mock_factory2, self.registry._registry["factory1"])

    def test_create_evaluator(self) -> None:
        """Test creating an evaluator instance from a registered factory."""
        # Arrange
        self.registry.register("factory1", self.mock_factory1)

        # Act
        evaluator = self.registry.create("factory1")

        # Assert
        self.mock_factory1.assert_called_once_with()
        self.assertEqual(self.mock_factory1.return_value, evaluator)

    def test_create_evaluator_with_kwargs(self) -> None:
        """Test creating an evaluator with keyword arguments."""
        # Arrange
        self.registry.register("factory1", self.mock_factory1)

        # Act
        evaluator = self.registry.create("factory1", param1="value1", param2=42)

        # Assert
        self.mock_factory1.assert_called_once_with(param1="value1", param2=42)
        self.assertEqual(self.mock_factory1.return_value, evaluator)

    def test_create_nonexistent_evaluator(self) -> None:
        """Test behavior when trying to create a non-existent evaluator."""
        # Act & Assert
        with self.assertRaises(KeyError):
            self.registry.create("nonexistent")

    def test_real_factory_functions(self) -> None:
        """Test with real factory functions rather than mocks."""

        # Arrange
        def create_mock_evaluator(**kwargs: Any) -> IEvaluator[Any, Any]:
            """Create a mock evaluator that always returns a fixed result."""
            evaluator = mock.MagicMock(spec=IEvaluator)
            evaluator.evaluate.return_value = EvaluationResult(
                is_correct=True,
                score=1.0,
                metadata=kwargs,
            )
            return evaluator

        def create_parametrized_evaluator(
            threshold: float = 0.5,
        ) -> IEvaluator[float, float]:
            """Create an evaluator with a configurable threshold."""
            evaluator = mock.MagicMock(spec=IEvaluator)
            evaluator.threshold = threshold
            evaluator.evaluate.return_value = EvaluationResult(
                is_correct=True,
                score=threshold,
            )
            return evaluator

        self.registry.register("mock_evaluator", create_mock_evaluator)
        self.registry.register("param_evaluator", create_parametrized_evaluator)

        # Act
        evaluator1 = self.registry.create("mock_evaluator", source="test")
        evaluator2 = self.registry.create("param_evaluator", threshold=0.8)

        # Assert
        self.assertTrue(isinstance(evaluator1, mock.MagicMock))
        self.assertTrue(isinstance(evaluator2, mock.MagicMock))

        result1 = evaluator1.evaluate("input", "expected")
        self.assertEqual({"source": "test"}, result1.metadata)

        self.assertEqual(0.8, evaluator2.threshold)


if __name__ == "__main__":
    unittest.main()
