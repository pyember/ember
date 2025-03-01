"""Unit tests for the CommonsenseQA dataset prepper."""

import unittest

from ember.core.utils.data.datasets_registry.commonsense_qa import (
    CommonsenseQAPrepper,
)


class TestCommonsenseQAPrepper(unittest.TestCase):
    """Test cases for the CommonsenseQAPrepper class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a prepper
        self.prepper = CommonsenseQAPrepper()

        # Create test data
        self.test_item = {
            "question": "What is the capital of France?",
            "choices": [
                {"label": "A", "text": "London"},
                {"label": "B", "text": "Paris"},
                {"label": "C", "text": "Berlin"},
                {"label": "D", "text": "Madrid"},
            ],
            "answerKey": "B",
        }

        self.test_item_malformed_choices = {
            "question": "What is the capital of France?",
            "choices": "not a list",
            "answerKey": "B",
        }

    def test_get_required_keys(self) -> None:
        """get_required_keys() should return the expected list of keys."""
        # Arrange & Act
        required_keys = self.prepper.get_required_keys()

        # Assert
        self.assertEqual(["question", "choices", "answerKey"], required_keys)

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
        self.assertEqual("B", entry.metadata["correct_answer"])

    def test_parse_choices(self) -> None:
        """_parse_choices() should extract choice labels and texts correctly."""
        # Arrange
        choices_data = self.test_item["choices"]

        # Act
        parsed_choices = self.prepper._parse_choices(choices_data=choices_data)

        # Assert
        expected_choices = {"A": "London", "B": "Paris", "C": "Berlin", "D": "Madrid"}
        self.assertEqual(expected_choices, parsed_choices)

    def test_parse_choices_not_list(self) -> None:
        """_parse_choices() should raise ValueError when choices is not a list."""
        # Arrange
        choices_data = "not a list"

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            self.prepper._parse_choices(choices_data=choices_data)

        self.assertIn("list", str(context.exception))

    def test_parse_choices_invalid_item(self) -> None:
        """_parse_choices() should gracefully handle invalid choice items."""
        # Arrange
        choices_data = [
            {"label": "A", "text": "London"},
            {"not_label": "B", "text": "Paris"},  # Missing "label" key
            {"label": "C", "not_text": "Berlin"},  # Missing "text" key
            "not a dict",  # Not a dictionary
        ]

        # Act
        parsed_choices = self.prepper._parse_choices(choices_data=choices_data)

        # Assert
        expected_choices = {"A": "London"}  # Only the valid choice should be included
        self.assertEqual(expected_choices, parsed_choices)

    def test_create_dataset_entries_malformed_choices(self) -> None:
        """create_dataset_entries() should raise ValueError for malformed choices."""
        # Arrange & Act & Assert
        with self.assertRaises(ValueError) as context:
            self.prepper.create_dataset_entries(item=self.test_item_malformed_choices)

        self.assertIn("list", str(context.exception))

    def test_create_dataset_entries_with_named_params(self) -> None:
        """create_dataset_entries() should require named parameters."""
        # Arrange & Act & Assert
        with self.assertRaises(TypeError):
            self.prepper.create_dataset_entries(self.test_item)  # type: ignore

    def test_parse_choices_with_named_params(self) -> None:
        """_parse_choices() should require named parameters."""
        # Arrange & Act & Assert
        with self.assertRaises(TypeError):
            self.prepper._parse_choices(self.test_item["choices"])  # type: ignore


if __name__ == "__main__":
    unittest.main()
