"""Unit tests for the TruthfulQA dataset prepper."""

import unittest
from typing import Dict, List, Any, Optional

from src.ember.core.utils.data.datasets_registry.truthful_qa import TruthfulQAPrepper
from src.ember.core.utils.data.base.models import DatasetEntry


class TestTruthfulQAPrepper(unittest.TestCase):
    """Test cases for the TruthfulQAPrepper class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a prepper
        self.prepper = TruthfulQAPrepper()

        # Create test data
        self.test_item = {
            "question": "What is the capital of France?",
            "mc1_targets": {
                "choices": ["London", "Paris", "Berlin", "Madrid"],
                "labels": [0, 1, 0, 0],  # Paris (index 1) is correct
            },
        }

        self.test_item_no_correct_answer = {
            "question": "What is the capital of France?",
            "mc1_targets": {
                "choices": ["London", "Berlin", "Madrid"],
                "labels": [0, 0, 0],  # No correct answer
            },
        }

    def test_get_required_keys(self) -> None:
        """get_required_keys() should return the expected list of keys."""
        # Arrange & Act
        required_keys = self.prepper.get_required_keys()

        # Assert
        self.assertEqual(["question", "mc1_targets"], required_keys)

    def test_create_dataset_entries(self) -> None:
        """create_dataset_entries() should handle properly formatted items."""
        # Arrange & Act
        entries = self.prepper.create_dataset_entries(item=self.test_item)

        # Assert
        self.assertEqual(1, len(entries))
        entry = entries[0]

        # Check entry content
        self.assertEqual(self.test_item["question"], entry.query)

        # Check choices mapping
        expected_choices = {"A": "London", "B": "Paris", "C": "Berlin", "D": "Madrid"}
        self.assertEqual(expected_choices, entry.choices)

        # Check metadata
        self.assertEqual(
            "B", entry.metadata["correct_answer"]
        )  # Should be B for Paris (index 1)

    def test_create_dataset_entries_no_correct_answer(self) -> None:
        """create_dataset_entries() should handle items with no correct answer."""
        # Arrange & Act
        entries = self.prepper.create_dataset_entries(
            item=self.test_item_no_correct_answer
        )

        # Assert
        self.assertEqual(1, len(entries))
        entry = entries[0]

        # Check entry content
        self.assertEqual(self.test_item_no_correct_answer["question"], entry.query)

        # Check choices mapping
        expected_choices = {"A": "London", "B": "Berlin", "C": "Madrid"}
        self.assertEqual(expected_choices, entry.choices)

        # Check metadata - should have None for correct_answer
        self.assertIsNone(entry.metadata["correct_answer"])

    def test_create_dataset_entries_type_conversion(self) -> None:
        """create_dataset_entries() should handle type conversion properly."""
        # Arrange
        test_item_with_strings = {
            "question": 42,  # Non-string question
            "mc1_targets": {
                "choices": ["London", "Paris", "Berlin", "Madrid"],
                "labels": [0, 1, 0, 0],
            },
        }

        # Act
        entries = self.prepper.create_dataset_entries(item=test_item_with_strings)

        # Assert
        self.assertEqual(1, len(entries))
        entry = entries[0]

        # Check that question was converted to string
        self.assertEqual("42", entry.query)
        self.assertTrue(isinstance(entry.query, str))

    def test_create_dataset_entries_with_named_params(self) -> None:
        """create_dataset_entries() should require named parameters."""
        # Arrange & Act & Assert
        with self.assertRaises(TypeError):
            self.prepper.create_dataset_entries(self.test_item)  # type: ignore


if __name__ == "__main__":
    unittest.main()
