"""Evaluators for numeric answer questions.

This module provides evaluators for numeric answer questions, such as
those found in math competitions like AIME.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Pattern, Tuple, Type

from ember.core.utils.eval.base_evaluator import EvaluationResult, IEvaluator


class IAnswerExtractor(ABC):
    """Interface for answer extraction strategies.

    Defines a consistent interface for extracting potential answers from text
    using various strategies. This follows the Strategy Pattern to decouple
    extraction algorithms from the evaluator.
    """

    @abstractmethod
    def extract(self, text: str) -> Tuple[List[int], Dict[str, Any]]:
        """Extract potential numeric answers from text.

        Args:
            text: The text to extract answers from

        Returns:
            A tuple containing:
              - A list of extracted integer values
              - Metadata about the extraction process
        """
        pass


class RegexAnswerExtractor(IAnswerExtractor):
    """Extracts answers using regular expressions.

    Base implementation for regex-based extractors that follow a consistent pattern.
    """

    def __init__(self, pattern: str, flags: int = re.IGNORECASE) -> None:
        """Initialize with a regex pattern.

        Args:
            pattern: Regular expression with capturing groups for answers
            flags: Regex flags (defaults to case-insensitive matching)
        """
        self.pattern = re.compile(pattern, flags)
        self.name = self.__class__.__name__

    def extract(self, text: str) -> Tuple[List[int], Dict[str, Any]]:
        """Extract numeric answers using the regex pattern.

        Args:
            text: The text to extract answers from

        Returns:
            Tuple of (extracted integers, metadata)
        """
        matches = self.pattern.findall(text)
        valid_numbers: List[int] = []

        # Handle both single group and multi-group patterns
        for match in matches:
            # If match is a tuple (multiple groups), check each group
            if isinstance(match, tuple):
                for group in match:
                    if group and group.strip():
                        try:
                            valid_numbers.append(int(group))
                        except ValueError:
                            pass
            # Otherwise, match is a single string
            elif match and match.strip():
                try:
                    valid_numbers.append(int(match))
                except ValueError:
                    pass

        return valid_numbers, {"method": self.name, "matches": matches}


class FinalAnswerExtractor(RegexAnswerExtractor):
    """Extracts answers from explicit 'final answer' statements."""

    def __init__(self) -> None:
        """Initialize with patterns for final answer statements."""
        # Make sure we require 'final answer' in the pattern, not optional
        pattern = r"(?:final\s+answer\s*(?:is|:|=))\s*(\d{1,3})"
        super().__init__(pattern)


class TheAnswerExtractor(RegexAnswerExtractor):
    """Extracts answers from 'the answer is' statements."""

    def __init__(self) -> None:
        """Initialize with patterns for 'the answer is' statements."""
        pattern = r"(?:the\s+answer\s*(?:is|:|=))\s*(\d{1,3})"
        super().__init__(pattern)


class EqualsExtractor(RegexAnswerExtractor):
    """Extracts answers from equals statements."""

    def __init__(self) -> None:
        """Initialize with patterns for equals statements."""
        pattern = r"(?:=\s*)(\d{1,3})"
        super().__init__(pattern)


class ThereforeExtractor(RegexAnswerExtractor):
    """Extracts answers from 'therefore' statements."""

    def __init__(self) -> None:
        """Initialize with patterns for 'therefore' statements."""
        pattern = r"(?:therefore,?\s+(?:the\s+)?answer\s*(?:is|:|=))\s*(\d{1,3})"
        super().__init__(pattern)


class GetAnswerExtractor(RegexAnswerExtractor):
    """Extracts answers from statements like 'we get X as our answer'."""

    def __init__(self) -> None:
        """Initialize with patterns for 'get X as our answer' statements."""
        pattern = r"(?:we|I)\s+get\s+(\d{1,3})\s+as\s+(?:our|the)\s+(?:final\s+)?answer"
        super().__init__(pattern)


class GenericNumberExtractor(RegexAnswerExtractor):
    """Extracts all numbers in the AIME range (0-999)."""

    def __init__(self) -> None:
        """Initialize with pattern for any numbers in AIME range."""
        pattern = r"(\d{1,3})"
        super().__init__(pattern)

    def extract(self, text: str) -> Tuple[List[int], Dict[str, Any]]:
        """Extract all numbers, filtering to AIME range.

        Args:
            text: The text to extract answers from

        Returns:
            Tuple of (filtered integers, metadata)
        """
        numbers, metadata = super().extract(text)

        # Filter to valid AIME range (0-999)
        valid_numbers = [num for num in numbers if 0 <= num <= 999]

        metadata["original_count"] = len(numbers)
        metadata["valid_count"] = len(valid_numbers)

        return valid_numbers, metadata


class NumericAnswerEvaluator(IEvaluator[str, str]):
    """Evaluator for exact numeric answers.

    Extracts numeric values from text responses and compares with expected answers.
    """

    def __init__(self, extract_pattern: Optional[str] = None) -> None:
        """Initialize with an optional custom extraction pattern.

        Args:
            extract_pattern: Regex pattern to extract numeric answers from text.
                Defaults to a pattern matching numbers with optional context.
        """
        # Use a simple extractor for basic number extraction
        if extract_pattern:
            self.extractor = RegexAnswerExtractor(extract_pattern)
        else:
            # Default extractor for general number extraction
            default_pattern = r"(?:answer|result)?\s*(?:is|=|:)?\s*(-?\d+)"
            self.extractor = RegexAnswerExtractor(default_pattern)

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: Any
    ) -> EvaluationResult:
        """Compare extracted numeric answer against expected value.

        Args:
            system_output: Text response containing a numeric answer
            correct_answer: Expected numeric answer as a string
            **kwargs: Additional keyword arguments (unused)

        Returns:
            EvaluationResult with correctness flag and score (1.0 or 0.0)
        """
        # Normalize and validate the expected answer
        try:
            expected = int(correct_answer.strip())
        except ValueError:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                metadata={"error": "Invalid reference answer format"},
            )

        # Extract numbers from the response
        extracted_numbers, metadata = self.extractor.extract(system_output)

        # Check if the expected number is among the extracted numbers
        is_correct = expected in extracted_numbers

        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            metadata={**metadata, "expected": expected, "found": is_correct},
        )


class AIMEAnswerEvaluator(IEvaluator[str, str]):
    """Specialized evaluator for AIME competition problems.

    AIME (American Invitational Mathematics Examination) answers are always
    integers in the range 0-999. This evaluator uses multiple extraction strategies
    in priority order to identify the intended answer in the response.
    """

    def __init__(
        self, custom_extractors: Optional[List[IAnswerExtractor]] = None
    ) -> None:
        """Initialize with AIME-specific extraction strategies.

        Args:
            custom_extractors: Optional list of custom extractors to use instead of defaults.
                If provided, these will be used in order. If not, default extractors will be used.
        """
        # Define extractors in priority order
        self.primary_extractors = custom_extractors or [
            FinalAnswerExtractor(),
            TheAnswerExtractor(),
            ThereforeExtractor(),
            GetAnswerExtractor(),
            EqualsExtractor(),
        ]

        # Fallback extractor for when no specific answer statements are found
        self.fallback_extractor = GenericNumberExtractor()

    def validate_aime_answer(self, answer_str: str) -> Tuple[bool, int, Optional[str]]:
        """Validate that an answer meets AIME format requirements.

        Args:
            answer_str: The answer string to validate

        Returns:
            Tuple of (is_valid, normalized_value, error_message)
        """
        try:
            value = int(answer_str.strip())
            if not (0 <= value <= 999):
                return False, value, f"AIME answers must be between 0-999, got {value}"
            return True, value, None
        except ValueError:
            return False, 0, f"Invalid AIME answer format: {answer_str}"

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: Any
    ) -> EvaluationResult:
        """Evaluate if the response contains the correct AIME answer.

        Uses multiple extraction strategies in priority order:
        1. Try each primary extractor to find answers in specific formats
        2. If no answers found, fall back to extracting all numbers

        Args:
            system_output: Model's response to the AIME problem
            correct_answer: Expected answer as string (0-999)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            EvaluationResult with correctness flag and detailed metadata
        """
        # Validate and normalize the expected answer
        is_valid, expected, error = self.validate_aime_answer(correct_answer)
        if not is_valid:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                metadata={"error": error, "expected": correct_answer},
            )

        # Try each primary extractor in sequence
        for extractor in self.primary_extractors:
            numbers, metadata = extractor.extract(system_output)

            # If we found potential answers with this extractor
            if numbers:
                is_correct = expected in numbers
                return EvaluationResult(
                    is_correct=is_correct,
                    score=1.0 if is_correct else 0.0,
                    metadata={
                        "extracted_method": "final_pattern",  # Maintain backward compatibility
                        "extractor": metadata["method"],
                        "extracted_values": numbers,
                        "expected": expected,
                    },
                )

        # If no primary extractors found answers, fall back to all numbers
        numbers, metadata = self.fallback_extractor.extract(system_output)

        is_correct = expected in numbers
        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            metadata={
                "extracted_method": "fallback_pattern",
                "extracted_values": numbers,
                "expected": expected,
            },
        )
