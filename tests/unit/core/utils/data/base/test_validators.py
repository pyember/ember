"""Unit tests for dataset validators."""

import unittest
from typing import Dict, List, Any, Optional
from unittest import mock

from datasets import Dataset, DatasetDict
from src.ember.core.utils.data.base.validators import (
    IDatasetValidator,
    DatasetValidator,
)


class TestIDatasetValidator(unittest.TestCase):
    """Test cases for the IDatasetValidator interface."""

    def test_interface_enforcement(self) -> None:
        """IDatasetValidator should require implementation of abstract methods."""
        # Attempt to instantiate the abstract base class directly
        with self.assertRaises(TypeError):
            IDatasetValidator()  # type: ignore

        # Create a subclass that doesn't implement any methods
        class IncompleteValidator(IDatasetValidator):
            pass

        # Attempt to instantiate the incomplete subclass
        with self.assertRaises(TypeError):
            IncompleteValidator()  # type: ignore

        # Create a subclass that implements only one method
        class PartialValidator1(IDatasetValidator):
            def validate_structure(self, *, dataset: Any) -> Any:
                return dataset

        # Attempt to instantiate with only one method
        with self.assertRaises(TypeError):
            PartialValidator1()  # type: ignore

        # Create a subclass that implements only two methods
        class PartialValidator2(IDatasetValidator):
            def validate_structure(self, *, dataset: Any) -> Any:
                return dataset

            def validate_required_keys(
                self, *, item: Dict[str, Any], required_keys: List[str]
            ) -> None:
                pass

        # Attempt to instantiate with only two methods
        with self.assertRaises(TypeError):
            PartialValidator2()  # type: ignore

        # Create a proper implementation
        class CompleteValidator(IDatasetValidator):
            def validate_structure(self, *, dataset: Any) -> Any:
                return dataset

            def validate_required_keys(
                self, *, item: Dict[str, Any], required_keys: List[str]
            ) -> None:
                pass

            def validate_item(
                self, *, item: Dict[str, Any], required_keys: List[str]
            ) -> None:
                pass

        # Should instantiate without error
        validator = CompleteValidator()
        self.assertIsInstance(validator, IDatasetValidator)


class TestDatasetValidator(unittest.TestCase):
    """Test cases for the DatasetValidator class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.validator = DatasetValidator()

        # Create test data
        self.list_data = [{"id": i, "value": f"test{i}"} for i in range(3)]

        # Create mock Dataset
        self.mock_dataset = mock.MagicMock(spec=Dataset)
        self.mock_dataset.__len__.return_value = 3

        # Create mock DatasetDict with proper spec
        self.mock_dataset_dict = mock.MagicMock(spec=DatasetDict)
        self.mock_dataset_dict.keys.return_value = ["train", "validation", "test"]
        self.mock_dataset_dict.__contains__.side_effect = lambda key: key in [
            "train",
            "validation",
            "test",
        ]
        self.mock_dataset_dict.__iter__.return_value = iter(
            ["train", "validation", "test"]
        )

        # Create mock splits
        self.mock_train_split = mock.MagicMock(spec=Dataset)
        self.mock_train_split.__len__.return_value = 10

        self.mock_validation_split = mock.MagicMock(spec=Dataset)
        self.mock_validation_split.__len__.return_value = 5

        self.mock_test_split = mock.MagicMock(spec=Dataset)
        self.mock_test_split.__len__.return_value = 7

        # Configure mock DatasetDict to return mock splits
        self.mock_dataset_dict.__getitem__.side_effect = lambda key: {
            "train": self.mock_train_split,
            "validation": self.mock_validation_split,
            "test": self.mock_test_split,
        }[key]

    def test_validate_structure_dataset(self) -> None:
        """validate_structure() should return the Dataset unchanged if it's non-empty."""
        # Arrange & Act
        result = self.validator.validate_structure(dataset=self.mock_dataset)

        # Assert
        self.assertIs(self.mock_dataset, result)

    def test_validate_structure_dataset_empty(self) -> None:
        """validate_structure() should raise ValueError if Dataset is empty."""
        # Arrange
        empty_dataset = mock.MagicMock(spec=Dataset)
        empty_dataset.__len__.return_value = 0

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            self.validator.validate_structure(dataset=empty_dataset)

        self.assertIn("empty", str(context.exception).lower())

    def test_validate_structure_dataset_dict_prefer_validation(self) -> None:
        """validate_structure() should prefer the 'validation' split from a DatasetDict."""
        # Arrange & Act
        result = self.validator.validate_structure(dataset=self.mock_dataset_dict)

        # Assert
        self.assertIs(self.mock_validation_split, result)

    def test_validate_structure_dataset_dict_fallback(self) -> None:
        """validate_structure() should fall back to first split when 'validation' is absent."""
        # Arrange
        # Create a mock DatasetDict without a 'validation' split
        mock_dataset_dict = mock.MagicMock(spec=DatasetDict)
        # Mock keys method
        mock_dataset_dict.keys.return_value = ["train", "test"]
        mock_dataset_dict.__contains__.side_effect = lambda key: key in [
            "train",
            "test",
        ]
        mock_dataset_dict.__iter__.return_value = iter(["train", "test"])

        mock_dataset_dict.__getitem__.side_effect = lambda key: {
            "train": self.mock_train_split,
            "test": self.mock_test_split,
        }[key]

        # Act
        result = self.validator.validate_structure(dataset=mock_dataset_dict)

        # Assert
        self.assertIs(self.mock_train_split, result)

    def test_validate_structure_dataset_dict_empty(self) -> None:
        """validate_structure() should raise ValueError if DatasetDict is empty."""
        # Arrange
        # Create an empty DatasetDict mock
        empty_dataset_dict = mock.MagicMock(spec=DatasetDict)
        # Return empty keys list
        empty_dataset_dict.keys.return_value = []

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            self.validator.validate_structure(dataset=empty_dataset_dict)

        self.assertIn("empty", str(context.exception).lower())

    def test_validate_structure_dataset_dict_empty_split(self) -> None:
        """validate_structure() should raise ValueError if the selected split is empty."""
        # Arrange
        # Configure a validation split that's empty
        self.mock_validation_split.__len__.return_value = 0

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            self.validator.validate_structure(dataset=self.mock_dataset_dict)

        self.assertIn("empty", str(context.exception).lower())
        self.assertIn("split", str(context.exception).lower())

    def test_validate_structure_list(self) -> None:
        """validate_structure() should return the list unchanged if it's non-empty."""
        # Arrange & Act
        result = self.validator.validate_structure(dataset=self.list_data)

        # Assert
        self.assertIs(self.list_data, result)

    def test_validate_structure_empty_list(self) -> None:
        """validate_structure() should raise ValueError if the list is empty."""
        # Arrange
        empty_list = []

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            self.validator.validate_structure(dataset=empty_list)

        self.assertIn("empty", str(context.exception).lower())

    def test_validate_structure_unsupported_type(self) -> None:
        """validate_structure() should raise TypeError for unsupported dataset types."""
        # Arrange
        unsupported_dataset = "not a valid dataset"

        # Act & Assert
        with self.assertRaises(TypeError) as context:
            self.validator.validate_structure(dataset=unsupported_dataset)

        self.assertIn("type", str(context.exception).lower())

    def test_validate_required_keys_all_present(self) -> None:
        """validate_required_keys() should not raise an error when all keys are present."""
        # Arrange
        item = {"id": 1, "question": "test", "answer": "test"}
        required_keys = ["id", "question"]

        # Act - Should not raise
        self.validator.validate_required_keys(item=item, required_keys=required_keys)

    def test_validate_required_keys_missing(self) -> None:
        """validate_required_keys() should raise ValueError when keys are missing."""
        # Arrange
        item = {"id": 1, "answer": "test"}
        required_keys = ["id", "question", "answer"]

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            self.validator.validate_required_keys(
                item=item, required_keys=required_keys
            )

        self.assertIn("missing", str(context.exception).lower())
        self.assertIn("question", str(context.exception))

    def test_validate_item_valid(self) -> None:
        """validate_item() should not raise an error when item is valid."""
        # Arrange
        item = {"id": 1, "question": "test", "answer": "test"}
        required_keys = ["id", "question"]

        # Act - Should not raise
        self.validator.validate_item(item=item, required_keys=required_keys)

    def test_validate_item_not_dict(self) -> None:
        """validate_item() should raise TypeError when item is not a dict."""
        # Arrange
        item = "not a dict"
        required_keys = ["id", "question"]

        # Act & Assert
        with self.assertRaises(TypeError) as context:
            self.validator.validate_item(item=item, required_keys=required_keys)  # type: ignore

        self.assertIn("dict", str(context.exception).lower())

    def test_validate_item_missing_key(self) -> None:
        """validate_item() should raise KeyError when a required key is missing."""
        # Arrange
        item = {"id": 1}
        required_keys = ["id", "question"]

        # Act & Assert
        with self.assertRaises(KeyError) as context:
            self.validator.validate_item(item=item, required_keys=required_keys)

        self.assertIn("missing", str(context.exception).lower())
        self.assertIn("question", str(context.exception))

    def test_validate_item_none_value(self) -> None:
        """validate_item() should raise KeyError when a required key has None value."""
        # Arrange
        item = {"id": 1, "question": None}
        required_keys = ["id", "question"]

        # Act & Assert
        with self.assertRaises(KeyError) as context:
            self.validator.validate_item(item=item, required_keys=required_keys)

        self.assertIn("none", str(context.exception).lower())
        self.assertIn("question", str(context.exception))

    def test_validate_with_named_parameters(self) -> None:
        """Validator methods should require named parameters."""
        # Arrange
        item = {"id": 1, "question": "test"}
        required_keys = ["id", "question"]

        # Act & Assert - For validate_structure
        with self.assertRaises(TypeError):
            self.validator.validate_structure(self.list_data)  # type: ignore

        # Act & Assert - For validate_required_keys
        with self.assertRaises(TypeError):
            self.validator.validate_required_keys(item, required_keys)  # type: ignore

        # Act & Assert - For validate_item
        with self.assertRaises(TypeError):
            self.validator.validate_item(item, required_keys)  # type: ignore


if __name__ == "__main__":
    unittest.main()
