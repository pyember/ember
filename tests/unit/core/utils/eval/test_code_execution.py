"""Unit tests for code execution evaluators.

These tests focus on the static behavior of the code execution evaluators,
such as initialization and error handling, avoiding tests that require 
actual code execution which is better tested in integration tests.
"""

from typing import Dict, List

from ember.core.utils.eval.code_execution import CodeCompetitionEvaluator, PythonHandler


class TestCodeCompetitionEvaluator:
    """Test suite for the competitive programming evaluator."""

    def test_initialization(self):
        """Test initialization with different parameters."""
        # Test with default parameters
        evaluator1 = CodeCompetitionEvaluator()
        assert evaluator1.time_limit == 2.0
        assert evaluator1.memory_limit_mb == 512
        assert evaluator1.supported_languages == ["python"]

        # Test with custom parameters
        evaluator2 = CodeCompetitionEvaluator(
            time_limit=5.0, memory_limit_mb=1024, supported_languages=["python", "cpp"]
        )
        assert evaluator2.time_limit == 5.0
        assert evaluator2.memory_limit_mb == 1024
        assert "python" in evaluator2.supported_languages
        assert "cpp" in evaluator2.supported_languages

    def test_error_handling(self):
        """Test handling of error cases."""
        evaluator = CodeCompetitionEvaluator()

        # Test unsupported language
        test_case = {"test_cases": [{"input": "5 7", "output": "12"}]}

        result = evaluator.evaluate(
            "print('test')", test_case, language="unsupported_lang"
        )
        assert result.is_correct is False
        assert result.score == 0.0
        assert "Unsupported language" in str(result.metadata)

        # Test empty test cases
        empty_test_case = {"test_cases": []}

        result = evaluator.evaluate("print('test')", empty_test_case, language="python")
        assert result.is_correct is False
        assert result.score == 0.0
        assert "No test cases" in str(result.metadata)

    # Note: Tests that require actual code execution are done in integration tests
    # Tests for python_execution, timeout_handling, test_case_validation, and score_calculation
    # are covered in tests/integration/core/utils/data/test_new_datasets_integration.py
