"""Fixtures and utilities for eval module tests."""

import sys
import os
from typing import Any, Dict, List

import pytest

# Print current path for debugging
print(f"Unit test Python path: {sys.path}")
print(f"Unit test current directory: {os.getcwd()}")

# Try both import paths
try:
    from ember.core.utils.eval.base_evaluator import EvaluationResult, IEvaluator
except ImportError:
    print("Trying alternative import path in unit tests...")
    try:
        from ember.core.utils.eval.base_evaluator import (
            EvaluationResult,
            IEvaluator,
        )
    except ImportError as e:
        print(f"Unit test import error: {e}")
        raise

try:
    from ember.core.utils.eval.evaluators import (
        ExactMatchEvaluator,
        NumericToleranceEvaluator,
        PartialRegexEvaluator,
    )
except ImportError:
    from ember.core.utils.eval.evaluators import (
        ExactMatchEvaluator,
        NumericToleranceEvaluator,
        PartialRegexEvaluator,
    )


@pytest.fixture
def evaluation_results() -> List[EvaluationResult]:
    """Return a list of sample EvaluationResult objects."""
    return [
        EvaluationResult(is_correct=True, score=1.0),
        EvaluationResult(is_correct=True, score=0.8, metadata={"confidence": 0.9}),
        EvaluationResult(is_correct=False, score=0.3, metadata={"confidence": 0.5}),
        EvaluationResult(is_correct=False, score=0.0, metadata={"error": "No match"}),
    ]


@pytest.fixture
def exact_match_evaluator() -> ExactMatchEvaluator:
    """Return an ExactMatchEvaluator instance."""
    return ExactMatchEvaluator()


@pytest.fixture
def numeric_tolerance_evaluator(tolerance: float = 0.01) -> NumericToleranceEvaluator:
    """Return a NumericToleranceEvaluator instance with specified tolerance."""
    return NumericToleranceEvaluator(tolerance=tolerance)


@pytest.fixture
def sample_qa_dataset() -> List[Dict[str, str]]:
    """Return a sample question-answering dataset."""
    return [
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


class MockExtractor:
    """Mock extractor for testing."""

    def __init__(self, return_value: str) -> None:
        """Initialize with a fixed return value."""
        self.return_value = return_value
        self.extract_called = False

    def extract(self, system_output: str, **kwargs: Any) -> str:
        """Return the fixed value regardless of input."""
        self.extract_called = True
        self.last_input = system_output
        self.last_kwargs = kwargs
        return self.return_value


class MockEvaluator(IEvaluator[Any, Any]):
    """Mock evaluator for testing."""

    def __init__(self, result: EvaluationResult) -> None:
        """Initialize with a fixed result to return."""
        self.result = result
        self.evaluate_called = False

    def evaluate(
        self, system_output: Any, correct_answer: Any, **kwargs: Any
    ) -> EvaluationResult:
        """Return the fixed result regardless of inputs."""
        self.evaluate_called = True
        self.last_output = system_output
        self.last_answer = correct_answer
        self.last_kwargs = kwargs
        return self.result
