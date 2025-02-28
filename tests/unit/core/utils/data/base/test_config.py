"""Unit tests for the BaseDatasetConfig class."""

import unittest
from typing import Optional

from src.ember.core.utils.data.base.config import BaseDatasetConfig


class TestBaseDatasetConfig(unittest.TestCase):
    """Test cases for the BaseDatasetConfig class.

    These tests verify the initialization, default values, and serialization
    behavior of the BaseDatasetConfig class, which serves as the base for
    all dataset configuration classes.
    """

    def test_init_with_defaults(self) -> None:
        """BaseDatasetConfig should use default values when none are provided."""
        # Arrange & Act
        config: BaseDatasetConfig = BaseDatasetConfig()

        # Assert
        self.assertIsNone(config.config_name)
        self.assertEqual("train", config.split)

    def test_init_with_custom_values(self) -> None:
        """BaseDatasetConfig should use provided values during initialization."""
        # Arrange
        custom_config_name: str = "custom_config"
        custom_split: str = "validation"

        # Act
        config: BaseDatasetConfig = BaseDatasetConfig(
            config_name=custom_config_name, split=custom_split
        )

        # Assert
        self.assertEqual(custom_config_name, config.config_name)
        self.assertEqual(custom_split, config.split)

    def test_model_serialization(self) -> None:
        """BaseDatasetConfig should correctly serialize to and deserialize from JSON."""
        # Arrange
        original_config: BaseDatasetConfig = BaseDatasetConfig(
            config_name="test_config", split="test"
        )

        # Act
        json_str: str = original_config.model_dump_json()
        deserialized_config: BaseDatasetConfig = BaseDatasetConfig.model_validate_json(
            json_str
        )

        # Assert
        self.assertEqual(original_config.config_name, deserialized_config.config_name)
        self.assertEqual(original_config.split, deserialized_config.split)

    def test_model_partial_initialization(self) -> None:
        """BaseDatasetConfig should handle partial initialization correctly."""
        # Arrange & Act
        config_with_name_only: BaseDatasetConfig = BaseDatasetConfig(
            config_name="only_name"
        )
        config_with_split_only: BaseDatasetConfig = BaseDatasetConfig(
            split="only_split"
        )

        # Assert
        self.assertEqual("only_name", config_with_name_only.config_name)
        self.assertEqual("train", config_with_name_only.split)

        self.assertIsNone(config_with_split_only.config_name)
        self.assertEqual("only_split", config_with_split_only.split)


if __name__ == "__main__":
    unittest.main()
