"""Tests for extractors module."""

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
    from ember.core.utils.eval.extractors import IOutputExtractor, RegexExtractor
except ImportError:
    from src.ember.core.utils.eval.extractors import IOutputExtractor, RegexExtractor


class TestIOutputExtractor(unittest.TestCase):
    """Tests for the IOutputExtractor interface."""

    def test_abstract_class_cannot_be_instantiated(self) -> None:
        """Test that IOutputExtractor cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            _ = IOutputExtractor()  # type: ignore

    def test_must_implement_extract(self) -> None:
        """Test that concrete subclasses must implement extract method."""

        # Arrange - Create a class that inherits but doesn't implement extract
        class IncompleteExtractor(IOutputExtractor[str, str]):
            pass

        # Act & Assert - Should raise TypeError when instantiating
        with self.assertRaises(TypeError):
            _ = IncompleteExtractor()  # type: ignore

    def test_concrete_implementation(self) -> None:
        """Test a concrete implementation with the extract method."""

        # Arrange - Create a simple concrete implementation
        class SimpleExtractor(IOutputExtractor[str, int]):
            def extract(self, system_output: str, **kwargs: Any) -> int:
                """Count the number of words in the output."""
                return len(system_output.split())

        # Act
        extractor = SimpleExtractor()
        result = extractor.extract("Hello world from Python")

        # Assert
        self.assertEqual(4, result)


class TestRegexExtractor(unittest.TestCase):
    """Tests for the RegexExtractor class."""

    def test_successful_extraction(self) -> None:
        """Test successful extraction of a pattern match."""
        # Arrange
        extractor = RegexExtractor(pattern=r"answer is (\w+)")

        # Act
        result = extractor.extract("The answer is Paris")

        # Assert
        self.assertEqual("Paris", result)

    def test_no_match(self) -> None:
        """Test behavior when the pattern doesn't match."""
        # Arrange
        extractor = RegexExtractor(pattern=r"answer is (\w+)")

        # Act
        result = extractor.extract("The response is Paris")

        # Assert
        self.assertEqual("", result)

    def test_multiple_matches_returns_first(self) -> None:
        """Test that only the first match is returned when multiple exist."""
        # Arrange
        extractor = RegexExtractor(pattern=r"answer is (\w+)")

        # Act
        result = extractor.extract("The answer is Paris. Another answer is London.")

        # Assert
        self.assertEqual("Paris", result)

    def test_match_with_multiple_groups(self) -> None:
        """Test behavior with a pattern containing multiple capture groups."""
        # Arrange
        extractor = RegexExtractor(pattern=r"(\w+) is (\w+)")

        # Act
        result = extractor.extract("The answer is Paris")

        # Assert - Should return only the first group
        self.assertEqual("answer", result)

    def test_greedy_matching(self) -> None:
        """Test greedy matching behavior."""
        # Arrange
        extractor = RegexExtractor(pattern=r"answer is (.*)")

        # Act
        result = extractor.extract("The answer is Paris, the capital of France")

        # Assert
        self.assertEqual("Paris, the capital of France", result)

    def test_non_greedy_matching(self) -> None:
        """Test non-greedy matching behavior."""
        # Arrange
        extractor = RegexExtractor(pattern=r"answer is (.*?)[\.,]")

        # Act
        result = extractor.extract("The answer is Paris, the capital of France")

        # Assert
        self.assertEqual("Paris", result)

    def test_case_sensitive_matching(self) -> None:
        """Test case sensitivity in pattern matching."""
        # Arrange
        extractor = RegexExtractor(pattern=r"Answer is (\w+)")  # Capital A

        # Act
        result1 = extractor.extract("The Answer is Paris")
        result2 = extractor.extract("The answer is Paris")  # Lowercase a

        # Assert
        self.assertEqual("Paris", result1)
        self.assertEqual("", result2)  # No match due to case difference

    def test_case_insensitive_matching(self) -> None:
        """Test case-insensitive pattern matching."""
        # Arrange
        extractor = RegexExtractor(
            pattern=r"(?i)answer is (\w+)"
        )  # Case-insensitive flag

        # Act
        result1 = extractor.extract("The Answer is Paris")
        result2 = extractor.extract("The answer is Paris")

        # Assert
        self.assertEqual("Paris", result1)
        self.assertEqual("Paris", result2)

    def test_special_characters(self) -> None:
        """Test matching with special characters in the pattern and input."""
        # Arrange
        extractor = RegexExtractor(pattern=r"result: ([\d\.\-]+)")

        # Act
        result = extractor.extract("The result: -123.45 is a negative number")

        # Assert
        self.assertEqual("-123.45", result)

    def test_empty_input(self) -> None:
        """Test behavior with an empty input string."""
        # Arrange
        extractor = RegexExtractor(pattern=r"answer is (\w+)")

        # Act
        result = extractor.extract("")

        # Assert
        self.assertEqual("", result)

    def test_whitespace_handling(self) -> None:
        """Test whitespace handling in regex patterns."""
        # Arrange
        extractor = RegexExtractor(pattern=r"answer\s+is\s+(\w+)")

        # Act - Test various whitespace configurations
        result1 = extractor.extract("The answer is Paris")
        result2 = extractor.extract("The answer  is Paris")
        result3 = extractor.extract("The answer is  Paris")
        result4 = extractor.extract("The answer\tis\nParis")

        # Assert
        self.assertEqual("Paris", result1)
        self.assertEqual("Paris", result2)
        self.assertEqual("Paris", result3)
        self.assertEqual("Paris", result4)


if __name__ == "__main__":
    unittest.main()
