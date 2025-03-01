"""Unit tests for the MMLU dataset prepper."""

import unittest

from ember.core.utils.data.datasets_registry.mmlu import MMLUPrepper, MMLUConfig


class TestMMLUConfig(unittest.TestCase):
    """Test cases for the MMLUConfig class."""

    def test_init_default_values(self) -> None:
        """MMLUConfig should initialize with default values."""
        # Arrange & Act
        config = MMLUConfig()

        # Assert
        self.assertIsNone(config.config_name)
        self.assertIsNone(config.split)

    def test_init_custom_values(self) -> None:
        """MMLUConfig should initialize with custom values."""
        # Arrange
        config_name = "abstract_algebra"
        split = "dev"

        # Act
        config = MMLUConfig(config_name=config_name, split=split)

        # Assert
        self.assertEqual(config_name, config.config_name)
        self.assertEqual(split, config.split)

    def test_model_serialization(self) -> None:
        """MMLUConfig should serialize to and deserialize from JSON correctly."""
        # Arrange
        original = MMLUConfig(config_name="abstract_algebra", split="dev")

        # Act
        json_str = original.model_dump_json()
        deserialized = MMLUConfig.model_validate_json(json_str)

        # Assert
        self.assertEqual(original.config_name, deserialized.config_name)
        self.assertEqual(original.split, deserialized.split)


class TestMMLUPrepper(unittest.TestCase):
    """Test cases for the MMLUPrepper class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a default prepper
        self.default_prepper = MMLUPrepper()

        # Create a prepper with custom config
        self.config = MMLUConfig(config_name="test_config", split="test_split")
        self.configured_prepper = MMLUPrepper(config=self.config)

        # Create test data
        self.test_item_with_int_answer = {
            "question": "What is the capital of France?",
            "choices": ["London", "Paris", "Berlin", "Madrid"],
            "answer": 1,  # Index-based answer (Paris)
        }

        self.test_item_with_str_answer = {
            "question": "What is the capital of France?",
            "choices": ["London", "Paris", "Berlin", "Madrid"],
            "answer": "B",  # Letter-based answer (Paris)
        }

        self.test_item_with_subject = {
            "question": "What is the capital of France?",
            "choices": ["London", "Paris", "Berlin", "Madrid"],
            "answer": 1,
            "subject": "geography",
        }

    def test_init_with_config(self) -> None:
        """MMLUPrepper should initialize with the provided config."""
        # Assert
        self.assertEqual(self.config, self.configured_prepper._config)
        self.assertEqual(self.config.config_name, self.configured_prepper.config_name)
        self.assertEqual(self.config.split, self.configured_prepper.split)

    def test_init_without_config(self) -> None:
        """MMLUPrepper should create a default config when none is provided."""
        # Assert
        self.assertIsInstance(self.default_prepper._config, MMLUConfig)
        self.assertIsNone(self.default_prepper.config_name)
        self.assertIsNone(self.default_prepper.split)

    def test_get_required_keys(self) -> None:
        """get_required_keys() should return the expected list of keys."""
        # Arrange & Act
        required_keys = self.default_prepper.get_required_keys()

        # Assert
        self.assertEqual(["question", "choices", "answer"], required_keys)
        # Verify "subject" is not required, as it's handled gracefully
        self.assertNotIn("subject", required_keys)

    def test_create_dataset_entries_int_answer(self) -> None:
        """create_dataset_entries() should handle items with integer answers."""
        # Arrange & Act
        entries = self.default_prepper.create_dataset_entries(
            item=self.test_item_with_int_answer
        )

        # Assert
        self.assertEqual(1, len(entries))
        entry = entries[0]

        # Check entry content
        self.assertEqual(self.test_item_with_int_answer["question"], entry.query)

        # Check choices mapping (should map to A, B, C, D)
        expected_choices = {
            "A": self.test_item_with_int_answer["choices"][0],
            "B": self.test_item_with_int_answer["choices"][1],
            "C": self.test_item_with_int_answer["choices"][2],
            "D": self.test_item_with_int_answer["choices"][3],
        }
        self.assertEqual(expected_choices, entry.choices)

        # Check metadata
        self.assertEqual(
            "B", entry.metadata["correct_answer"]
        )  # Should convert 1 to "B"
        self.assertIsNone(entry.metadata["subject"])
        self.assertIsNone(entry.metadata["config_name"])

    def test_create_dataset_entries_str_answer(self) -> None:
        """create_dataset_entries() should handle items with string answers."""
        # Arrange & Act
        entries = self.default_prepper.create_dataset_entries(
            item=self.test_item_with_str_answer
        )

        # Assert
        self.assertEqual(1, len(entries))
        entry = entries[0]

        # Check metadata
        self.assertEqual("B", entry.metadata["correct_answer"])  # Should keep "B" as is

    def test_create_dataset_entries_with_subject(self) -> None:
        """create_dataset_entries() should handle items with a subject field."""
        # Arrange & Act
        entries = self.default_prepper.create_dataset_entries(
            item=self.test_item_with_subject
        )

        # Assert
        self.assertEqual(1, len(entries))
        entry = entries[0]

        # Check metadata
        self.assertEqual("geography", entry.metadata["subject"])

    def test_create_dataset_entries_with_config(self) -> None:
        """create_dataset_entries() should include config_name in metadata."""
        # Arrange & Act
        entries = self.configured_prepper.create_dataset_entries(
            item=self.test_item_with_int_answer
        )

        # Assert
        self.assertEqual(1, len(entries))
        entry = entries[0]

        # Check metadata
        self.assertEqual("test_config", entry.metadata["config_name"])

    def test_create_dataset_entries_with_named_params(self) -> None:
        """create_dataset_entries() should require named parameters."""
        # Arrange & Act & Assert
        with self.assertRaises(TypeError):
            self.default_prepper.create_dataset_entries(self.test_item_with_int_answer)  # type: ignore


if __name__ == "__main__":
    unittest.main()
