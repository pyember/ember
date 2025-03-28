"""Unit tests for numeric answer evaluators."""

import pytest

from ember.core.utils.eval.numeric_answer import (
    NumericAnswerEvaluator,
    AIMEAnswerEvaluator,
)


class TestNumericAnswerEvaluator:
    """Test suite for the numeric answer evaluator."""

    def test_exact_match(self):
        """Test exact match of numeric answers."""
        evaluator = NumericAnswerEvaluator()

        # Simple exact match
        result = evaluator.evaluate("The answer is 42", "42")
        assert result.is_correct is True
        assert result.score == 1.0

        # No match
        result = evaluator.evaluate("The answer is 43", "42")
        assert result.is_correct is False
        assert result.score == 0.0

    def test_extract_from_text(self):
        """Test extracting numbers from text responses."""
        evaluator = NumericAnswerEvaluator()

        # Extract from beginning of text
        result = evaluator.evaluate("42 is the answer to everything", "42")
        assert result.is_correct is True

        # Extract from middle of text
        result = evaluator.evaluate("The answer is 42 and that's final", "42")
        assert result.is_correct is True

        # Extract from end of text
        result = evaluator.evaluate("After much calculation, I got 42", "42")
        assert result.is_correct is True

        # Extract with equals sign
        result = evaluator.evaluate("x = 42", "42")
        assert result.is_correct is True

        # No numbers in text
        result = evaluator.evaluate("The answer is unknown", "42")
        assert result.is_correct is False

    def test_custom_pattern(self):
        """Test using custom regex patterns."""
        # Custom pattern to extract only numbers after "Final answer:"
        evaluator = NumericAnswerEvaluator(extract_pattern=r"Final answer:\s*(\d+)")

        # Should match
        result = evaluator.evaluate("Final answer: 42", "42")
        assert result.is_correct is True

        # Should not match numbers not following the pattern
        result = evaluator.evaluate("The answer is 42", "42")
        assert result.is_correct is False

    def test_invalid_reference_answer(self):
        """Test handling invalid reference answers."""
        evaluator = NumericAnswerEvaluator()

        # Non-numeric reference answer
        result = evaluator.evaluate("The answer is 42", "abc")
        assert result.is_correct is False
        assert "error" in result.metadata


class TestAIMEAnswerEvaluator:
    """Test suite for the AIME-specific answer evaluator."""

    def test_valid_aime_answers(self):
        """Test evaluation of valid AIME answers (0-999)."""
        evaluator = AIMEAnswerEvaluator()

        # Simple valid answers
        result = evaluator.evaluate("The answer is 42", "42")
        assert result.is_correct is True

        result = evaluator.evaluate("The answer is 007", "7")
        assert result.is_correct is True

        result = evaluator.evaluate("The answer is 999", "999")
        assert result.is_correct is True

        # Invalid answers (out of range)
        result = evaluator.evaluate("The answer is 1000", "1000")
        assert result.is_correct is False
        assert "error" in result.metadata

    def test_extract_from_working(self):
        """Test extracting answers from solution working."""
        evaluator = AIMEAnswerEvaluator()

        # Extract from calculation steps
        long_solution = """
        First, we need to calculate x = 10.
        Then y = 20.
        Finally, x + y + z = 42.
        Therefore, the answer is 42.
        """
        result = evaluator.evaluate(long_solution, "42")
        assert result.is_correct is True
        assert result.metadata["extracted_method"] == "final_pattern"

    def test_final_answer_pattern(self):
        """Test priority of final answer pattern extraction."""
        evaluator = AIMEAnswerEvaluator()

        # Should extract from final answer statement even with other numbers present
        response = "I calculated 10, 25, and 30, but the final answer is 42."
        result = evaluator.evaluate(response, "42")
        assert result.is_correct is True
        assert result.metadata["extracted_method"] == "final_pattern"

        # Should correctly handle different final answer formats
        cases = [
            "Final answer: 42",
            "The answer is 42",
            "Therefore, the answer = 42",
            "We get 42 as our final answer",
            "The final answer is 42",
        ]

        for case in cases:
            result = evaluator.evaluate(case, "42")
            assert result.is_correct is True
            assert result.metadata["extracted_method"] == "final_pattern"

    def test_fallback_extraction(self):
        """Test fallback to all numbers when no final answer statement is found."""
        evaluator = AIMEAnswerEvaluator()

        # Should find the correct number among others when no final answer statement
        response = "I calculated 10, 25, 42, and 30."
        result = evaluator.evaluate(response, "42")
        assert result.is_correct is True
        assert result.metadata["extracted_method"] == "fallback_pattern"

        # Should handle three-digit numbers correctly
        response = "The values are 100, 250, and 375."
        result = evaluator.evaluate(response, "375")
        assert result.is_correct is True
        assert result.metadata["extracted_method"] == "fallback_pattern"

        # No matching number
        response = "The values are 100, 250, and 375."
        result = evaluator.evaluate(response, "42")
        assert result.is_correct is False

    def test_invalid_reference_format(self):
        """Test handling of invalid reference answer formats."""
        evaluator = AIMEAnswerEvaluator()

        # Non-numeric reference
        result = evaluator.evaluate("The answer is 42", "not_a_number")
        assert result.is_correct is False
        assert "error" in result.metadata

        # Out of range reference
        result = evaluator.evaluate("The answer is 1234", "1234")
        assert result.is_correct is False
        assert "error" in result.metadata

    def test_custom_pattern(self):
        """Test with custom extractors."""
        # Create a custom extractor for specific patterns
        from ember.core.utils.eval.numeric_answer import RegexAnswerExtractor

        custom_extractor = RegexAnswerExtractor(r"answer:\s*(\d{1,3})")
        evaluator = AIMEAnswerEvaluator(custom_extractors=[custom_extractor])

        # Should match the custom pattern
        result = evaluator.evaluate("My answer: 42", "42")
        assert result.is_correct is True
        assert result.metadata["extracted_method"] == "final_pattern"

        # Should not match patterns that don't fit the custom regex
        result = evaluator.evaluate("The answer is 42", "42")
        assert result.is_correct is True  # Still correct but uses fallback
        assert result.metadata["extracted_method"] == "fallback_pattern"
