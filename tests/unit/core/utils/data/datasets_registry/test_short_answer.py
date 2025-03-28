"""Unit tests for the ShortAnswer dataset prepper."""

import unittest

from ember.core.utils.data.datasets_registry.short_answer import ShortAnswerPrepper


class TestShortAnswerPrepper(unittest.TestCase):
    """Test cases for the ShortAnswerPrepper class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a prepper
        self.prepper = ShortAnswerPrepper()

        # Create test data
        self.test_item = {
            "question": "What is the capital of France?",
            "answer": "Paris",
        }

        self.test_item_numbers = {
            "question": 42,  # Non-string question
            "answer": 123,  # Non-string answer
        }

    def test_get_required_keys(self) -> None:
        """get_required_keys() should return the expected list of keys."""
        # Arrange & Act
        required_keys = self.prepper.get_required_keys()

        # Assert
        self.assertEqual(["question", "answer"], required_keys)

    def test_create_dataset_entries(self) -> None:
        """create_dataset_entries() should handle properly formatted items."""
        # Arrange & Act
        entries = self.prepper.create_dataset_entries(item=self.test_item)

        # Assert
        self.assertEqual(1, len(entries))
        entry = entries[0]

        # Check entry content
        self.assertEqual(self.test_item["question"], entry.query)

        # Check that choices is empty
        self.assertEqual({}, entry.choices)

        # Check metadata
        self.assertEqual(self.test_item["answer"], entry.metadata["gold_answer"])

    def test_create_dataset_entries_type_conversion(self) -> None:
        """create_dataset_entries() should handle type conversion properly."""
        # Arrange & Act
        entries = self.prepper.create_dataset_entries(item=self.test_item_numbers)

        # Assert
        self.assertEqual(1, len(entries))
        entry = entries[0]

        # Check that question and answer were converted to strings
        self.assertEqual("42", entry.query)
        self.assertTrue(isinstance(entry.query, str))

        self.assertEqual("123", entry.metadata["gold_answer"])
        self.assertTrue(isinstance(entry.metadata["gold_answer"], str))

    def test_create_dataset_entries_with_named_params(self) -> None:
        """create_dataset_entries() should require named parameters."""
        # Arrange & Act & Assert
        with self.assertRaises(TypeError):
            self.prepper.create_dataset_entries(self.test_item)  # type: ignore


if __name__ == "__main__":
    unittest.main()
