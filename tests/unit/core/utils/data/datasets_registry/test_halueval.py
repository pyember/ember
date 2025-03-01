"""Unit tests for the HaluEval dataset prepper."""

import unittest

from ember.core.utils.data.datasets_registry.halueval import (
    HaluEvalPrepper,
    HaluEvalConfig,
)


class TestHaluEvalConfig(unittest.TestCase):
    """Test cases for the HaluEvalConfig class."""

    def test_init_default_values(self) -> None:
        """HaluEvalConfig should initialize with default values."""
        # Arrange & Act
        config = HaluEvalConfig()

        # Assert
        self.assertEqual("qa", config.config_name)
        self.assertEqual("data", config.split)

    def test_init_custom_values(self) -> None:
        """HaluEvalConfig should initialize with custom values."""
        # Arrange
        config_name = "custom_config"
        split = "custom_split"

        # Act
        config = HaluEvalConfig(config_name=config_name, split=split)

        # Assert
        self.assertEqual(config_name, config.config_name)
        self.assertEqual(split, config.split)

    def test_model_serialization(self) -> None:
        """HaluEvalConfig should serialize to and deserialize from JSON correctly."""
        # Arrange
        original = HaluEvalConfig(config_name="custom_config", split="custom_split")

        # Act
        json_str = original.model_dump_json()
        deserialized = HaluEvalConfig.model_validate_json(json_str)

        # Assert
        self.assertEqual(original.config_name, deserialized.config_name)
        self.assertEqual(original.split, deserialized.split)


class TestHaluEvalPrepper(unittest.TestCase):
    """Test cases for the HaluEvalPrepper class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a prepper
        self.prepper = HaluEvalPrepper()

        # Create test data
        self.test_item = {
            "knowledge": "Paris is the capital of France.",
            "question": "What is the capital of France?",
            "right_answer": "Paris is the capital of France.",
            "hallucinated_answer": "London is the capital of France.",
        }

    def test_get_required_keys(self) -> None:
        """get_required_keys() should return the expected list of keys."""
        # Arrange & Act
        required_keys = self.prepper.get_required_keys()

        # Assert
        self.assertEqual(
            ["knowledge", "question", "right_answer", "hallucinated_answer"],
            required_keys,
        )

    def test_create_dataset_entries(self) -> None:
        """create_dataset_entries() should create two entries for each item."""
        # Arrange & Act
        entries = self.prepper.create_dataset_entries(item=self.test_item)

        # Assert
        self.assertEqual(2, len(entries))

        # Check first entry (not hallucinated)
        not_hallucinated_entry = entries[0]
        self.assertIn(self.test_item["knowledge"], not_hallucinated_entry.query)
        self.assertIn(self.test_item["question"], not_hallucinated_entry.query)
        self.assertIn(self.test_item["right_answer"], not_hallucinated_entry.query)
        self.assertEqual(
            {"A": "Not Hallucinated", "B": "Hallucinated"},
            not_hallucinated_entry.choices,
        )
        self.assertEqual("A", not_hallucinated_entry.metadata["correct_answer"])

        # Check second entry (hallucinated)
        hallucinated_entry = entries[1]
        self.assertIn(self.test_item["knowledge"], hallucinated_entry.query)
        self.assertIn(self.test_item["question"], hallucinated_entry.query)
        self.assertIn(self.test_item["hallucinated_answer"], hallucinated_entry.query)
        self.assertEqual(
            {"A": "Not Hallucinated", "B": "Hallucinated"}, hallucinated_entry.choices
        )
        self.assertEqual("B", hallucinated_entry.metadata["correct_answer"])

    def test_build_dataset_entry(self) -> None:
        """_build_dataset_entry() should format the query and metadata correctly."""
        # Arrange
        knowledge = "Paris is the capital of France."
        question = "What is the capital of France?"
        candidate_answer = "Paris is the capital of France."
        correct_choice = "A"

        # Act
        entry = self.prepper._build_dataset_entry(
            knowledge=knowledge,
            question=question,
            candidate_answer=candidate_answer,
            correct_choice=correct_choice,
        )

        # Assert
        expected_query = (
            f"Knowledge: {knowledge}\n"
            f"Question: {question}\n"
            f"Candidate Answer: {candidate_answer}. "
            "Is this candidate answer supported by the provided knowledge?"
        )
        self.assertEqual(expected_query, entry.query)
        self.assertEqual({"A": "Not Hallucinated", "B": "Hallucinated"}, entry.choices)
        self.assertEqual(correct_choice, entry.metadata["correct_answer"])

    def test_create_dataset_entries_with_named_params(self) -> None:
        """create_dataset_entries() should require named parameters."""
        # Arrange & Act & Assert
        with self.assertRaises(TypeError):
            self.prepper.create_dataset_entries(self.test_item)  # type: ignore

    def test_build_dataset_entry_with_named_params(self) -> None:
        """_build_dataset_entry() should require named parameters."""
        # Arrange & Act & Assert
        with self.assertRaises(TypeError):
            self.prepper._build_dataset_entry(
                "knowledge", "question", "answer", "A"  # type: ignore
            )


if __name__ == "__main__":
    unittest.main()
