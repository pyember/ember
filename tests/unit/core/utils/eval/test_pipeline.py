"""Tests for pipeline module."""

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
    from ember.core.utils.eval.pipeline import (
        PipelineEvaluator,
        BatchEvaluationSummary,
        evaluate_batch,
        evaluate_batch_with_summary,
        summarize_batch,
    )
except ImportError:
    from ember.core.utils.eval.pipeline import (
        PipelineEvaluator,
        BatchEvaluationSummary,
        evaluate_batch,
        evaluate_batch_with_summary,
        summarize_batch,
    )


class TestPipelineEvaluator(unittest.TestCase):
    """Tests for the PipelineEvaluator class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create transforms
        self.transform1 = mock.MagicMock(return_value="transformed1")
        self.transform2 = mock.MagicMock(return_value="transformed2")

        # Create mock evaluator
        self.mock_evaluator = mock.MagicMock(spec=IEvaluator)
        self.mock_evaluator.evaluate.return_value = EvaluationResult(
            is_correct=True, score=1.0
        )

        # Create pipeline evaluator
        self.pipeline = PipelineEvaluator(
            transforms=[self.transform1, self.transform2],
            evaluator=self.mock_evaluator,
        )

    def test_transform_chain(self) -> None:
        """Test that transforms are applied in sequence."""
        # Act
        self.pipeline.evaluate(
            system_output="raw_output",
            correct_answer="expected_answer",
        )

        # Assert
        self.transform1.assert_called_once_with("raw_output")
        self.transform2.assert_called_once_with("transformed1")
        self.mock_evaluator.evaluate.assert_called_once_with(
            "transformed2", "expected_answer"
        )

    def test_transform_chain_with_kwargs(self) -> None:
        """Test that keyword arguments are passed to the final evaluator."""
        # Act
        self.pipeline.evaluate(
            system_output="raw_output",
            correct_answer="expected_answer",
            custom_param=True,
        )

        # Assert
        self.mock_evaluator.evaluate.assert_called_once_with(
            "transformed2", "expected_answer", custom_param=True
        )

    def test_empty_transforms_list(self) -> None:
        """Test behavior with an empty transforms list."""
        # Arrange
        pipeline = PipelineEvaluator(
            transforms=[],
            evaluator=self.mock_evaluator,
        )

        # Act
        result = pipeline.evaluate(
            system_output="raw_output",
            correct_answer="expected_answer",
        )

        # Assert
        self.mock_evaluator.evaluate.assert_called_once_with(
            "raw_output", "expected_answer"
        )
        self.assertEqual(self.mock_evaluator.evaluate.return_value, result)

    def test_real_transforms(self) -> None:
        """Test with actual transformation functions."""

        # Arrange
        def to_upper(s: str) -> str:
            return s.upper()

        def reverse_string(s: str) -> str:
            return s[::-1]

        exact_match_evaluator = mock.MagicMock(spec=IEvaluator)
        exact_match_evaluator.evaluate.return_value = EvaluationResult(
            is_correct=True, score=1.0
        )

        pipeline = PipelineEvaluator(
            transforms=[to_upper, reverse_string],
            evaluator=exact_match_evaluator,
        )

        # Act
        pipeline.evaluate(
            system_output="hello",
            correct_answer="OLLEH",  # Expected result after to_upper and reverse
        )

        # Assert
        exact_match_evaluator.evaluate.assert_called_once_with("OLLEH", "OLLEH")


class TestSummarizeBatch(unittest.TestCase):
    """Tests for the summarize_batch function."""

    def test_empty_batch(self) -> None:
        """Test summarizing an empty batch."""
        # Act
        summary = summarize_batch([])

        # Assert
        self.assertEqual([], summary.results)
        self.assertEqual(0.0, summary.mean_score)
        self.assertEqual(0.0, summary.accuracy)

    def test_all_correct(self) -> None:
        """Test summarizing a batch where all results are correct."""
        # Arrange
        results = [
            EvaluationResult(is_correct=True, score=1.0),
            EvaluationResult(is_correct=True, score=0.8),
            EvaluationResult(is_correct=True, score=0.9),
        ]

        # Act
        summary = summarize_batch(results)

        # Assert
        self.assertEqual(results, summary.results)
        self.assertAlmostEqual(0.9, summary.mean_score)
        self.assertEqual(1.0, summary.accuracy)

    def test_mixed_results(self) -> None:
        """Test summarizing a batch with mixed results."""
        # Arrange
        results = [
            EvaluationResult(is_correct=True, score=1.0),
            EvaluationResult(is_correct=False, score=0.0),
            EvaluationResult(is_correct=True, score=0.8),
        ]

        # Act
        summary = summarize_batch(results)

        # Assert
        self.assertEqual(results, summary.results)
        self.assertAlmostEqual(0.6, summary.mean_score)
        self.assertAlmostEqual(2 / 3, summary.accuracy)

    def test_all_incorrect(self) -> None:
        """Test summarizing a batch where all results are incorrect."""
        # Arrange
        results = [
            EvaluationResult(is_correct=False, score=0.2),
            EvaluationResult(is_correct=False, score=0.4),
            EvaluationResult(is_correct=False, score=0.0),
        ]

        # Act
        summary = summarize_batch(results)

        # Assert
        self.assertEqual(results, summary.results)
        self.assertAlmostEqual(0.2, summary.mean_score)
        self.assertEqual(0.0, summary.accuracy)


class TestEvaluateBatch(unittest.TestCase):
    """Tests for the evaluate_batch function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.mock_evaluator = mock.MagicMock(spec=IEvaluator)
        self.mock_evaluator.evaluate.side_effect = [
            EvaluationResult(is_correct=True, score=1.0),
            EvaluationResult(is_correct=False, score=0.0),
            EvaluationResult(is_correct=True, score=0.8),
        ]

    def test_empty_batch(self) -> None:
        """Test evaluating an empty batch."""
        # Act
        results = evaluate_batch(
            evaluator=self.mock_evaluator,
            system_outputs=[],
            correct_answers=[],
        )

        # Assert
        self.assertEqual([], results)
        self.mock_evaluator.evaluate.assert_not_called()

    def test_batch_evaluation(self) -> None:
        """Test evaluating a batch of inputs."""
        # Arrange
        system_outputs = ["output1", "output2", "output3"]
        correct_answers = ["answer1", "answer2", "answer3"]

        # Act
        results = evaluate_batch(
            evaluator=self.mock_evaluator,
            system_outputs=system_outputs,
            correct_answers=correct_answers,
            custom_param=True,
        )

        # Assert
        self.assertEqual(3, len(results))
        self.assertTrue(results[0].is_correct)
        self.assertFalse(results[1].is_correct)
        self.assertTrue(results[2].is_correct)

        # Check that evaluate was called for each item with the correct parameters
        self.assertEqual(3, self.mock_evaluator.evaluate.call_count)
        self.mock_evaluator.evaluate.assert_has_calls(
            [
                mock.call("output1", "answer1", custom_param=True),
                mock.call("output2", "answer2", custom_param=True),
                mock.call("output3", "answer3", custom_param=True),
            ]
        )

    def test_mismatched_lengths(self) -> None:
        """Test behavior with mismatched input lists."""
        # Arrange
        system_outputs = ["output1", "output2"]
        correct_answers = ["answer1", "answer2", "answer3"]

        # Act & Assert
        with self.assertRaises(ValueError):
            evaluate_batch(
                evaluator=self.mock_evaluator,
                system_outputs=system_outputs,
                correct_answers=correct_answers,
            )


class TestEvaluateBatchWithSummary(unittest.TestCase):
    """Tests for the evaluate_batch_with_summary function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.mock_evaluator = mock.MagicMock(spec=IEvaluator)
        self.mock_evaluator.evaluate.side_effect = [
            EvaluationResult(is_correct=True, score=1.0),
            EvaluationResult(is_correct=False, score=0.0),
            EvaluationResult(is_correct=True, score=0.8),
        ]

    @mock.patch("ember.core.utils.eval.pipeline.evaluate_batch")
    @mock.patch("ember.core.utils.eval.pipeline.summarize_batch")
    def test_function_composition(
        self, mock_summarize: mock.MagicMock, mock_evaluate: mock.MagicMock
    ) -> None:
        """Test that the function composes evaluate_batch and summarize_batch."""
        # Arrange
        system_outputs = ["output1", "output2", "output3"]
        correct_answers = ["answer1", "answer2", "answer3"]

        # Mock return values
        mock_results = [
            EvaluationResult(is_correct=True, score=1.0),
            EvaluationResult(is_correct=False, score=0.0),
        ]
        mock_summary = BatchEvaluationSummary(
            results=mock_results,
            mean_score=0.5,
            accuracy=0.5,
        )

        mock_evaluate.return_value = mock_results
        mock_summarize.return_value = mock_summary

        # Act
        result = evaluate_batch_with_summary(
            evaluator=self.mock_evaluator,
            system_outputs=system_outputs,
            correct_answers=correct_answers,
            custom_param=True,
        )

        # Assert
        mock_evaluate.assert_called_once_with(
            evaluator=self.mock_evaluator,
            system_outputs=system_outputs,
            correct_answers=correct_answers,
            custom_param=True,
        )
        mock_summarize.assert_called_once_with(mock_results)
        self.assertEqual(mock_summary, result)

    def test_end_to_end(self) -> None:
        """Test the function from end to end without mocking."""
        # Arrange
        system_outputs = ["output1", "output2", "output3"]
        correct_answers = ["answer1", "answer2", "answer3"]

        # Act
        summary = evaluate_batch_with_summary(
            evaluator=self.mock_evaluator,
            system_outputs=system_outputs,
            correct_answers=correct_answers,
        )

        # Assert
        self.assertEqual(3, len(summary.results))
        self.assertAlmostEqual(0.6, summary.mean_score)
        self.assertAlmostEqual(2 / 3, summary.accuracy)


if __name__ == "__main__":
    unittest.main()
