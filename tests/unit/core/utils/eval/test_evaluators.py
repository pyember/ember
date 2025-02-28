"""Tests for evaluators module."""

import re
import subprocess
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
    from ember.core.utils.eval.evaluators import (
        ComposedEvaluator,
        ExactMatchEvaluator,
        NumericToleranceEvaluator,
        PartialRegexEvaluator,
        CodeExecutionEvaluator,
    )
except ImportError:
    from src.ember.core.utils.eval.evaluators import (
        ComposedEvaluator,
        ExactMatchEvaluator,
        NumericToleranceEvaluator,
        PartialRegexEvaluator,
        CodeExecutionEvaluator,
    )

try:
    from ember.core.utils.eval.extractors import RegexExtractor
except ImportError:
    from src.ember.core.utils.eval.extractors import RegexExtractor


class TestComposedEvaluator(unittest.TestCase):
    """Tests for the ComposedEvaluator class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a mock extractor
        self.mock_extractor = mock.MagicMock()

        # Create a mock base evaluator
        self.mock_base_evaluator = mock.MagicMock(spec=IEvaluator)

        # Create the composed evaluator with the mocks
        self.evaluator = ComposedEvaluator(
            extractor=self.mock_extractor,
            base_evaluator=self.mock_base_evaluator,
        )

    def test_evaluate_calls_extractor_and_evaluator(self) -> None:
        """Test that evaluate calls both the extractor and the base evaluator."""
        # Arrange
        self.mock_extractor.extract.return_value = "extracted_value"
        expected_result = EvaluationResult(is_correct=True, score=1.0)
        self.mock_base_evaluator.evaluate.return_value = expected_result

        # Act
        result = self.evaluator.evaluate(
            system_output="raw_output",
            correct_answer="expected_answer",
            custom_param=True,
        )

        # Assert
        self.mock_extractor.extract.assert_called_once_with(
            "raw_output", custom_param=True
        )
        self.mock_base_evaluator.evaluate.assert_called_once_with(
            "extracted_value", "expected_answer", custom_param=True
        )
        self.assertEqual(expected_result, result)

    def test_with_real_components(self) -> None:
        """Test ComposedEvaluator with real components."""

        # Arrange - A regex extractor and exact match evaluator
        class SimpleExtractor:
            def extract(self, text: str, **kwargs: Any) -> str:
                """Extract the first word from the text."""
                match = re.search(r"\b(\w+)\b", text)
                return match.group(1) if match else ""

        exact_evaluator = ExactMatchEvaluator()
        composed = ComposedEvaluator(
            extractor=SimpleExtractor(),
            base_evaluator=exact_evaluator,
        )

        # Act
        result = composed.evaluate("Hello world", "Hello")

        # Assert
        self.assertTrue(result.is_correct)
        self.assertEqual(1.0, result.score)


class TestExactMatchEvaluator(unittest.TestCase):
    """Tests for the ExactMatchEvaluator class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.evaluator = ExactMatchEvaluator()

    def test_exact_match(self) -> None:
        """Test exact string matches."""
        # Act
        result = self.evaluator.evaluate("hello", "hello")

        # Assert
        self.assertTrue(result.is_correct)
        self.assertEqual(1.0, result.score)

    def test_case_insensitive_match(self) -> None:
        """Test case-insensitive matches."""
        # Act
        result = self.evaluator.evaluate("Hello", "hello")

        # Assert
        self.assertTrue(result.is_correct)
        self.assertEqual(1.0, result.score)

    def test_whitespace_insensitive_match(self) -> None:
        """Test whitespace-insensitive matches."""
        # Act
        result = self.evaluator.evaluate("  hello  ", "hello")

        # Assert
        self.assertTrue(result.is_correct)
        self.assertEqual(1.0, result.score)

    def test_non_match(self) -> None:
        """Test non-matching strings."""
        # Act
        result = self.evaluator.evaluate("hello", "world")

        # Assert
        self.assertFalse(result.is_correct)
        self.assertEqual(0.0, result.score)

    def test_empty_strings(self) -> None:
        """Test with empty strings."""
        # Act & Assert
        self.assertTrue(self.evaluator.evaluate("", "").is_correct)
        self.assertFalse(self.evaluator.evaluate("hello", "").is_correct)
        self.assertFalse(self.evaluator.evaluate("", "hello").is_correct)

    def test_special_characters(self) -> None:
        """Test with strings containing special characters."""
        # Act & Assert
        self.assertTrue(self.evaluator.evaluate("!@#$%^&*()", "!@#$%^&*()").is_correct)
        self.assertFalse(self.evaluator.evaluate("!@#$%^&*()", "!@#$%^&*").is_correct)

    def test_unicode_characters(self) -> None:
        """Test with strings containing Unicode characters."""
        # Act & Assert
        self.assertTrue(self.evaluator.evaluate("こんにちは", "こんにちは").is_correct)
        self.assertFalse(self.evaluator.evaluate("こんにちは", "さようなら").is_correct)


class TestNumericToleranceEvaluator(unittest.TestCase):
    """Tests for the NumericToleranceEvaluator class."""

    def test_exact_match(self) -> None:
        """Test exact numeric matches."""
        # Arrange
        evaluator = NumericToleranceEvaluator(tolerance=0.01)

        # Act
        result = evaluator.evaluate(system_output=10.0, correct_answer=10.0)

        # Assert
        self.assertTrue(result.is_correct)
        self.assertEqual(1.0, result.score)
        self.assertEqual({"diff": 0.0}, result.metadata)

    def test_within_tolerance(self) -> None:
        """Test values within the specified tolerance."""
        # Arrange
        evaluator = NumericToleranceEvaluator(tolerance=0.1)

        # Act
        result = evaluator.evaluate(system_output=10.05, correct_answer=10.0)

        # Assert
        self.assertTrue(result.is_correct)
        self.assertAlmostEqual(0.995, result.score, places=6)
        self.assertEqual({"diff": 0.05}, result.metadata)

    def test_outside_tolerance(self) -> None:
        """Test values outside the specified tolerance."""
        # Arrange
        evaluator = NumericToleranceEvaluator(tolerance=0.01)

        # Act
        result = evaluator.evaluate(system_output=10.05, correct_answer=10.0)

        # Assert
        self.assertFalse(result.is_correct)
        self.assertAlmostEqual(0.995, result.score, places=6)
        self.assertEqual({"diff": 0.05}, result.metadata)

    def test_negative_values(self) -> None:
        """Test with negative values."""
        # Arrange
        evaluator = NumericToleranceEvaluator(tolerance=0.1)

        # Act
        result = evaluator.evaluate(system_output=-10.05, correct_answer=-10.0)

        # Assert
        self.assertTrue(result.is_correct)
        self.assertAlmostEqual(0.995, result.score, places=6)
        self.assertEqual({"diff": 0.05}, result.metadata)

    def test_zero_correct_answer(self) -> None:
        """Test with zero as the correct answer to avoid division by zero."""
        # Arrange
        evaluator = NumericToleranceEvaluator(tolerance=0.1)

        # Act
        result = evaluator.evaluate(system_output=0.05, correct_answer=0.0)

        # Assert
        self.assertTrue(result.is_correct)
        self.assertAlmostEqual(0.95, result.score, places=6)
        self.assertEqual({"diff": 0.05}, result.metadata)

    def test_large_difference(self) -> None:
        """Test with a large difference between values."""
        # Arrange
        evaluator = NumericToleranceEvaluator(tolerance=0.1)

        # Act
        result = evaluator.evaluate(system_output=20.0, correct_answer=10.0)

        # Assert
        self.assertFalse(result.is_correct)
        self.assertEqual(0.0, result.score)  # Max clamped at 0.0
        self.assertEqual({"diff": 10.0}, result.metadata)

    def test_custom_tolerance(self) -> None:
        """Test with a custom tolerance value."""
        # Arrange
        evaluator = NumericToleranceEvaluator(tolerance=5.0)

        # Act
        result = evaluator.evaluate(system_output=14.0, correct_answer=10.0)

        # Assert
        self.assertTrue(result.is_correct)
        self.assertAlmostEqual(0.6, result.score, places=6)
        self.assertEqual({"diff": 4.0}, result.metadata)


class TestCodeExecutionEvaluator(unittest.TestCase):
    """Tests for the CodeExecutionEvaluator class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.evaluator = CodeExecutionEvaluator()

    @mock.patch("subprocess.run")
    def test_successful_execution_matching_output(
        self, mock_run: mock.MagicMock
    ) -> None:
        """Test successful code execution with matching output."""
        # Arrange - Mock successful execution
        mock_process = mock.MagicMock()
        mock_process.stdout = "Hello, World!\n"
        mock_process.stderr = ""
        mock_process.returncode = 0
        mock_run.return_value = mock_process

        # Act
        result = self.evaluator.evaluate(
            system_output='print("Hello, World!")', correct_answer="Hello, World!"
        )

        # Assert
        self.assertTrue(result.is_correct)
        self.assertEqual(1.0, result.score)
        self.assertEqual(
            {"stdout": "Hello, World!\n", "stderr": "", "exit_code": 0}, result.metadata
        )
        mock_run.assert_called_once_with(
            args=["python", "-c", 'print("Hello, World!")'],
            capture_output=True,
            text=True,
            timeout=5.0,
        )

    @mock.patch("subprocess.run")
    def test_successful_execution_non_matching_output(
        self, mock_run: mock.MagicMock
    ) -> None:
        """Test successful code execution with non-matching output."""
        # Arrange - Mock successful execution with different output
        mock_process = mock.MagicMock()
        mock_process.stdout = "Incorrect output\n"
        mock_process.stderr = ""
        mock_process.returncode = 0
        mock_run.return_value = mock_process

        # Act
        result = self.evaluator.evaluate(
            system_output='print("Incorrect output")', correct_answer="Expected output"
        )

        # Assert
        self.assertFalse(result.is_correct)
        self.assertEqual(0.0, result.score)
        self.assertEqual(
            {"stdout": "Incorrect output\n", "stderr": "", "exit_code": 0},
            result.metadata,
        )

    @mock.patch("subprocess.run")
    def test_execution_error(self, mock_run: mock.MagicMock) -> None:
        """Test execution with syntax error in the code."""
        # Arrange - Mock execution with error
        mock_process = mock.MagicMock()
        mock_process.stdout = ""
        mock_process.stderr = "SyntaxError: invalid syntax\n"
        mock_process.returncode = 1
        mock_run.return_value = mock_process

        # Act
        result = self.evaluator.evaluate(
            system_output='print("Hello, World!"',  # Missing closing parenthesis
            correct_answer="Hello, World!",
        )

        # Assert
        self.assertFalse(result.is_correct)
        self.assertEqual(0.0, result.score)
        self.assertEqual(
            {"stdout": "", "stderr": "SyntaxError: invalid syntax\n", "exit_code": 1},
            result.metadata,
        )

    @mock.patch("subprocess.run")
    def test_timeout_error(self, mock_run: mock.MagicMock) -> None:
        """Test execution that times out."""
        # Arrange - Mock a timeout exception
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="", timeout=5.0)

        # Act
        result = self.evaluator.evaluate(
            system_output="import time; time.sleep(10)",
            correct_answer="This will never happen",
        )

        # Assert
        self.assertFalse(result.is_correct)
        self.assertEqual(0.0, result.score)
        self.assertIn("error", result.metadata)
        self.assertIn("TimeoutExpired", result.metadata["error"])

    @mock.patch("subprocess.run")
    def test_custom_timeout(self, mock_run: mock.MagicMock) -> None:
        """Test with a custom timeout value."""
        # Arrange
        evaluator = CodeExecutionEvaluator(timeout=10.0)
        mock_process = mock.MagicMock()
        mock_process.stdout = "Hello, World!\n"
        mock_process.stderr = ""
        mock_process.returncode = 0
        mock_run.return_value = mock_process

        # Act
        evaluator.evaluate(
            system_output='print("Hello, World!")', correct_answer="Hello, World!"
        )

        # Assert - Check timeout was passed correctly
        mock_run.assert_called_once_with(
            args=["python", "-c", 'print("Hello, World!")'],
            capture_output=True,
            text=True,
            timeout=10.0,
        )


class TestPartialRegexEvaluator(unittest.TestCase):
    """Tests for the PartialRegexEvaluator class."""

    def test_matching_regex(self) -> None:
        """Test with a regex that matches the input."""
        # Arrange
        evaluator = PartialRegexEvaluator(pattern=r"answer is (\w+)")

        # Act
        result = evaluator.evaluate(
            system_output="The answer is Paris", correct_answer="Paris"
        )

        # Assert
        self.assertTrue(result.is_correct)
        self.assertEqual(1.0, result.score)

    def test_non_matching_regex(self) -> None:
        """Test with a regex that doesn't match the input."""
        # Arrange
        evaluator = PartialRegexEvaluator(pattern=r"answer is (\w+)")

        # Act
        result = evaluator.evaluate(
            system_output="The response is Paris", correct_answer="Paris"
        )

        # Assert
        self.assertFalse(result.is_correct)
        self.assertEqual(0.0, result.score)

    def test_matching_regex_wrong_answer(self) -> None:
        """Test with a regex that matches but extracts the wrong answer."""
        # Arrange
        evaluator = PartialRegexEvaluator(pattern=r"answer is (\w+)")

        # Act
        result = evaluator.evaluate(
            system_output="The answer is London", correct_answer="Paris"
        )

        # Assert
        self.assertFalse(result.is_correct)
        self.assertEqual(0.0, result.score)

    def test_case_sensitivity(self) -> None:
        """Test case sensitivity in the extracted value."""
        # Arrange
        evaluator = PartialRegexEvaluator(pattern=r"answer is (\w+)")

        # Act
        result = evaluator.evaluate(
            system_output="The answer is PARIS", correct_answer="paris"
        )

        # Assert - The ExactMatchEvaluator is case-insensitive
        self.assertTrue(result.is_correct)
        self.assertEqual(1.0, result.score)


if __name__ == "__main__":
    unittest.main()
