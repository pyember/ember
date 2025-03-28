"""Unit tests for the GPQA dataset prepper."""

import unittest

from ember.core.utils.data.datasets_registry.gpqa import GPQAConfig, GPQAPrepper


class TestGPQAConfig(unittest.TestCase):
    """Test cases for the GPQAConfig class."""

    def test_init_default_values(self) -> None:
        """GPQAConfig should initialize with default values."""
        # Arrange & Act
        config = GPQAConfig()

        # Assert
        self.assertEqual("gpqa_diamond", config.subset)
        self.assertIsNone(config.difficulty)
        self.assertIsNone(config.domain)

    def test_init_custom_values(self) -> None:
        """GPQAConfig should initialize with custom values."""
        # Arrange
        subset = "gpqa_diamond_2"
        difficulty = "hard"
        domain = "physics"

        # Act
        config = GPQAConfig(subset=subset, difficulty=difficulty, domain=domain)

        # Assert
        self.assertEqual(subset, config.subset)
        self.assertEqual(difficulty, config.difficulty)
        self.assertEqual(domain, config.domain)

    def test_model_serialization(self) -> None:
        """GPQAConfig should serialize to and deserialize from JSON correctly."""
        # Arrange
        original = GPQAConfig(
            subset="gpqa_diamond", difficulty="hard", domain="physics"
        )

        # Act
        json_str = original.model_dump_json()
        deserialized = GPQAConfig.model_validate_json(json_str)

        # Assert
        self.assertEqual(original.subset, deserialized.subset)
        self.assertEqual(original.difficulty, deserialized.difficulty)
        self.assertEqual(original.domain, deserialized.domain)


class TestGPQAPrepper(unittest.TestCase):
    """Test cases for the GPQAPrepper class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a default prepper
        self.default_prepper = GPQAPrepper()

        # Create a prepper with custom config
        self.config = GPQAConfig(
            subset="test_subset", difficulty="hard", domain="physics"
        )
        self.configured_prepper = GPQAPrepper(config=self.config)

        # Create test data
        self.test_item = {
            "question_id": "GPQA-123",
            "question": "What is the relationship between mass and energy in special relativity?",
            "choices": {
                "A": "E = mc",
                "B": "E = mc²",
                "C": "E = m/c²",
                "D": "E = m+c²",
            },
            "answer": "B",
            "domain": "physics",
            "difficulty": "medium",
        }

        # Create item with different domain
        self.test_item_chemistry = {
            "question_id": "GPQA-456",
            "question": "Which of the following is not an amino acid?",
            "choices": {
                "A": "Glycine",
                "B": "Alanine",
                "C": "Glucose",
                "D": "Tyrosine",
            },
            "answer": "C",
            "domain": "chemistry",
            "difficulty": "medium",
        }

    def test_init_with_config(self) -> None:
        """GPQAPrepper should initialize with the provided config."""
        # Assert
        self.assertEqual(self.config, self.configured_prepper._config)
        self.assertEqual(self.config.subset, self.configured_prepper.subset)
        self.assertEqual(self.config.difficulty, self.configured_prepper.difficulty)
        self.assertEqual(self.config.domain, self.configured_prepper.domain)

    def test_init_without_config(self) -> None:
        """GPQAPrepper should create a default config when none is provided."""
        # Assert
        self.assertIsInstance(self.default_prepper._config, GPQAConfig)
        self.assertEqual("gpqa_diamond", self.default_prepper.subset)
        self.assertIsNone(self.default_prepper.difficulty)
        self.assertIsNone(self.default_prepper.domain)

    def test_init_with_string_config(self) -> None:
        """GPQAPrepper should handle string config as subset name."""
        # Arrange & Act
        prepper = GPQAPrepper(config="custom_subset")

        # Assert
        self.assertEqual("custom_subset", prepper.subset)

    def test_get_required_keys(self) -> None:
        """get_required_keys() should return the expected list of keys."""
        # Arrange & Act
        required_keys = self.default_prepper.get_required_keys()

        # Assert
        expected_keys = [
            "question_id",
            "question",
            "choices",
            "answer",
            "domain",
            "difficulty",
        ]
        self.assertEqual(set(expected_keys), set(required_keys))

    def test_create_dataset_entries_basic(self) -> None:
        """create_dataset_entries() should process items correctly."""
        # Arrange & Act
        entries = self.default_prepper.create_dataset_entries(item=self.test_item)

        # Assert
        self.assertEqual(1, len(entries))
        entry = entries[0]

        # Check entry content
        self.assertEqual(self.test_item["question"], entry.query)
        self.assertEqual(self.test_item["choices"], entry.choices)

        # Check metadata
        self.assertEqual(self.test_item["answer"], entry.metadata["correct_answer"])
        self.assertEqual(self.test_item["question_id"], entry.metadata["question_id"])
        self.assertEqual(self.test_item["domain"], entry.metadata["domain"])
        self.assertEqual(self.test_item["difficulty"], entry.metadata["difficulty"])
        self.assertEqual("multiple_choice", entry.metadata["task_type"])
        self.assertEqual("gpqa", entry.metadata["dataset"])
        self.assertEqual("gpqa_diamond", entry.metadata["subset"])

    def test_create_dataset_entries_with_domain_filter(self) -> None:
        """create_dataset_entries() should filter by domain."""
        # Arrange
        prepper = GPQAPrepper(config=GPQAConfig(domain="physics"))

        # Act - Physics item should pass
        physics_entries = prepper.create_dataset_entries(item=self.test_item)
        # Act - Chemistry item should be filtered out
        chemistry_entries = prepper.create_dataset_entries(
            item=self.test_item_chemistry
        )

        # Assert
        self.assertEqual(1, len(physics_entries))
        self.assertEqual(0, len(chemistry_entries))

    def test_create_dataset_entries_with_difficulty_filter(self) -> None:
        """create_dataset_entries() should filter by difficulty."""
        # Arrange
        prepper = GPQAPrepper(config=GPQAConfig(difficulty="hard"))

        # Act - Medium difficulty item should be filtered out
        entries = prepper.create_dataset_entries(item=self.test_item)

        # Assert
        self.assertEqual(0, len(entries))

        # Modify the difficulty to match filter
        self.test_item["difficulty"] = "hard"
        entries = prepper.create_dataset_entries(item=self.test_item)
        self.assertEqual(1, len(entries))

    def test_create_dataset_entries_empty_fields(self) -> None:
        """create_dataset_entries() should handle missing optional fields gracefully."""
        # Arrange
        item_without_optionals = {
            "question_id": "GPQA-789",
            "question": "Test question?",
            "choices": {"A": "Option A", "B": "Option B"},
            "answer": "A",
        }

        # Act
        entries = self.default_prepper.create_dataset_entries(
            item=item_without_optionals
        )

        # Assert
        self.assertEqual(1, len(entries))
        entry = entries[0]
        self.assertEqual("", entry.metadata["domain"])
        self.assertEqual("", entry.metadata["difficulty"])


if __name__ == "__main__":
    unittest.main()
