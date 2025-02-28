"""Unit tests for the DatasetService class."""

import unittest
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest import mock

from src.ember.core.utils.data.service import DatasetService
from src.ember.core.utils.data.base.loaders import IDatasetLoader
from src.ember.core.utils.data.base.validators import IDatasetValidator
from src.ember.core.utils.data.base.samplers import IDatasetSampler
from src.ember.core.utils.data.base.transformers import IDatasetTransformer
from src.ember.core.utils.data.base.preppers import IDatasetPrepper
from src.ember.core.utils.data.base.models import DatasetEntry, DatasetInfo, TaskType
from src.ember.core.utils.data.base.config import BaseDatasetConfig


class TestDatasetService(unittest.TestCase):
    """Test cases for the DatasetService class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_loader = mock.MagicMock(spec=IDatasetLoader)
        self.mock_validator = mock.MagicMock(spec=IDatasetValidator)
        self.mock_sampler = mock.MagicMock(spec=IDatasetSampler)
        self.mock_transformer1 = mock.MagicMock(spec=IDatasetTransformer)
        self.mock_transformer2 = mock.MagicMock(spec=IDatasetTransformer)

        # Mock dataset info
        self.dataset_info = DatasetInfo(
            name="test_dataset",
            description="Test dataset",
            source="test_source",
            task_type=TaskType.MULTIPLE_CHOICE,
        )

        # Mock prepper
        self.mock_prepper = mock.MagicMock(spec=IDatasetPrepper)
        self.mock_prepper.get_required_keys.return_value = ["id", "question", "answer"]

        # Mock config
        self.config = BaseDatasetConfig(config_name="test_config", split="test_split")

        # Mock dataset items and entries
        self.mock_dataset_items = [
            {"id": 1, "question": "Q1", "answer": "A1"},
            {"id": 2, "question": "Q2", "answer": "A2"},
            {"id": 3, "question": "Q3", "answer": "A3"},
        ]

        self.mock_dataset_entries = [
            DatasetEntry(query=f"Q{i}", metadata={"id": i, "answer": f"A{i}"})
            for i in range(1, 4)
        ]

        # Mock raw dataset
        self.raw_dataset = mock.MagicMock()
        self.raw_dataset.__getitem__.side_effect = lambda i: self.mock_dataset_items[i]

        # Configure prepper mock
        self.mock_prepper.create_dataset_entries.side_effect = lambda *, item: [
            self.mock_dataset_entries[item["id"] - 1]
        ]

        # Configure loader mock
        self.mock_loader.load.return_value = self.raw_dataset

        # Configure validator mock
        self.mock_validator.validate_structure.return_value = self.raw_dataset

        # Configure transformer mocks
        self.mock_transformer1.transform.return_value = self.raw_dataset
        self.mock_transformer2.transform.return_value = self.raw_dataset

        # Configure sampler mock
        self.mock_sampler.sample.return_value = self.raw_dataset

        # Create service instance
        self.service = DatasetService(
            loader=self.mock_loader,
            validator=self.mock_validator,
            sampler=self.mock_sampler,
            transformers=[self.mock_transformer1, self.mock_transformer2],
        )

    def test_init(self) -> None:
        """DatasetService initialization should store provided components."""
        # Arrange & Act
        service = DatasetService(
            loader=self.mock_loader,
            validator=self.mock_validator,
            sampler=self.mock_sampler,
            transformers=[self.mock_transformer1, self.mock_transformer2],
        )

        # Assert
        self.assertEqual(self.mock_loader, service._loader)
        self.assertEqual(self.mock_validator, service._validator)
        self.assertEqual(self.mock_sampler, service._sampler)
        self.assertEqual(
            [self.mock_transformer1, self.mock_transformer2], service._transformers
        )

    def test_init_default_transformers(self) -> None:
        """DatasetService should initialize with an empty transformers list by default."""
        # Arrange & Act
        service = DatasetService(
            loader=self.mock_loader,
            validator=self.mock_validator,
            sampler=self.mock_sampler,
        )

        # Assert
        self.assertEqual([], service._transformers)

    def test_resolve_config_name_string(self) -> None:
        """_resolve_config_name() should return the string directly."""
        # Arrange
        config_str = "test_config"

        # Act
        result = self.service._resolve_config_name(config=config_str)

        # Assert
        self.assertEqual(config_str, result)

    def test_resolve_config_name_config_object(self) -> None:
        """_resolve_config_name() should return the config_name attribute from a config object."""
        # Arrange & Act
        result = self.service._resolve_config_name(config=self.config)

        # Assert
        self.assertEqual(self.config.config_name, result)

    def test_resolve_config_name_none(self) -> None:
        """_resolve_config_name() should return None when config is None."""
        # Arrange & Act
        result = self.service._resolve_config_name(config=None)

        # Assert
        self.assertIsNone(result)

    def test_load_data(self) -> None:
        """_load_data() should call the loader with appropriate parameters."""
        # Arrange
        dataset_name = "test_dataset"
        config_name = "test_config"

        # Act
        result = self.service._load_data(dataset_name=dataset_name, config=config_name)

        # Assert
        self.assertEqual(self.raw_dataset, result)
        self.mock_loader.load.assert_called_once_with(
            dataset_name=dataset_name, config=config_name
        )

    def test_select_split_with_config(self) -> None:
        """select_split() should select the split specified in the config object."""
        # Arrange
        mock_dataset = {"train": "train_data", "test_split": "test_data"}

        # Act
        result = self.service.select_split(dataset=mock_dataset, config_obj=self.config)

        # Assert
        self.assertEqual("test_data", result)

    def test_select_split_missing_split(self) -> None:
        """select_split() should return the original dataset when split is not found."""
        # Arrange
        mock_dataset = {"train": "train_data", "validation": "validation_data"}

        # Act
        result = self.service.select_split(dataset=mock_dataset, config_obj=self.config)

        # Assert
        self.assertEqual(mock_dataset, result)

    def test_select_split_no_config(self) -> None:
        """select_split() should return the original dataset when config is None."""
        # Arrange
        mock_dataset = {"train": "train_data", "test": "test_data"}

        # Act
        result = self.service.select_split(dataset=mock_dataset, config_obj=None)

        # Assert
        self.assertEqual(mock_dataset, result)

    def test_validate_structure(self) -> None:
        """_validate_structure() should call the validator with appropriate parameters."""
        # Arrange & Act
        result = self.service._validate_structure(dataset=self.raw_dataset)

        # Assert
        self.assertEqual(self.raw_dataset, result)
        self.mock_validator.validate_structure.assert_called_once_with(
            dataset=self.raw_dataset
        )

    def test_transform_data(self) -> None:
        """_transform_data() should apply all transformers in sequence."""
        # Arrange & Act
        result = self.service._transform_data(data=self.raw_dataset)

        # Assert
        self.assertEqual(self.raw_dataset, result)
        self.mock_transformer1.transform.assert_called_once_with(data=self.raw_dataset)
        self.mock_transformer2.transform.assert_called_once_with(data=self.raw_dataset)

    def test_validate_keys(self) -> None:
        """_validate_keys() should call the validator for a sample of dataset items."""
        # Arrange
        self.raw_dataset.__len__ = mock.MagicMock(
            return_value=len(self.mock_dataset_items)
        )

        # Act
        self.service._validate_keys(data=self.raw_dataset, prepper=self.mock_prepper)

        # Assert
        self.mock_prepper.get_required_keys.assert_called_once()
        # Should validate at least one item
        self.assertGreaterEqual(
            self.mock_validator.validate_required_keys.call_count, 1
        )

    def test_sample_data(self) -> None:
        """_sample_data() should call the sampler with appropriate parameters."""
        # Arrange
        num_samples = 2

        # Act
        result = self.service._sample_data(
            data=self.raw_dataset, num_samples=num_samples
        )

        # Assert
        self.assertEqual(self.raw_dataset, result)
        self.mock_sampler.sample.assert_called_once_with(
            data=self.raw_dataset, num_samples=num_samples
        )

    def test_prep_data(self) -> None:
        """_prep_data() should process each item with the prepper."""
        # Arrange
        # Configure the raw dataset to act like a sequence
        self.raw_dataset.__iter__ = mock.MagicMock(
            return_value=iter(self.mock_dataset_items)
        )

        # Act
        result = self.service._prep_data(
            dataset_info=self.dataset_info,
            sampled_data=self.raw_dataset,
            prepper=self.mock_prepper,
        )

        # Assert
        self.assertEqual(len(self.mock_dataset_items), len(result))
        self.assertEqual(self.mock_dataset_entries, result)

        # Verify prepper called for each item
        self.assertEqual(
            len(self.mock_dataset_items),
            self.mock_prepper.create_dataset_entries.call_count,
        )

        # Verify validation called for each item
        self.assertEqual(
            len(self.mock_dataset_items), self.mock_validator.validate_item.call_count
        )

    def test_prep_data_error_handling(self) -> None:
        """_prep_data() should handle errors for individual items and continue processing."""

        # Arrange
        # Mock validator to raise error for first item
        def validate_item_side_effect(
            *, item: Dict[str, Any], required_keys: List[str]
        ) -> None:
            if item["id"] == 1:
                raise ValueError("Test error")

        self.mock_validator.validate_item.side_effect = validate_item_side_effect

        # Configure the raw dataset to act like a sequence
        self.raw_dataset.__iter__ = mock.MagicMock(
            return_value=iter(self.mock_dataset_items)
        )

        # Act
        result = self.service._prep_data(
            dataset_info=self.dataset_info,
            sampled_data=self.raw_dataset,
            prepper=self.mock_prepper,
        )

        # Assert - should only have entries for items 2 and 3
        self.assertEqual(2, len(result))
        self.assertEqual(self.mock_dataset_entries[1:], result)

    def test_load_and_prepare_end_to_end(self) -> None:
        """load_and_prepare() should execute the complete processing pipeline."""
        # Arrange
        num_samples = 2

        # Override the mock calls to chain their results properly
        self.mock_loader.load.return_value = "loaded_dataset"

        def select_split_side_effect(
            dataset: Any, config_obj: Optional[BaseDatasetConfig]
        ) -> Any:
            self.assertEqual("loaded_dataset", dataset)
            return "split_dataset"

        # Use a test double for select_split
        original_select_split = self.service.select_split
        self.service.select_split = mock.MagicMock(side_effect=select_split_side_effect)

        self.mock_validator.validate_structure.side_effect = lambda *, dataset: {
            "split_dataset": "validated_dataset"
        }.get(dataset, dataset)

        self.mock_transformer1.transform.side_effect = lambda *, data: {
            "validated_dataset": "transformed1_dataset"
        }.get(data, data)

        self.mock_transformer2.transform.side_effect = lambda *, data: {
            "transformed1_dataset": "transformed2_dataset"
        }.get(data, data)

        self.mock_sampler.sample.side_effect = lambda *, data, num_samples: {
            "transformed2_dataset": "sampled_dataset"
        }.get(data, data)

        # Mock _validate_keys to avoid random sampling in test
        original_validate_keys = self.service._validate_keys
        self.service._validate_keys = mock.MagicMock()

        # Mock _prep_data to return simple result
        original_prep_data = self.service._prep_data
        mock_entries = [DatasetEntry(query="Q1"), DatasetEntry(query="Q2")]
        self.service._prep_data = mock.MagicMock(return_value=mock_entries)

        try:
            # Act
            result = self.service.load_and_prepare(
                dataset_info=self.dataset_info,
                prepper=self.mock_prepper,
                config=self.config,
                num_samples=num_samples,
            )

            # Assert
            self.assertEqual(mock_entries, result)

            # Verify correct sequence of calls
            self.mock_loader.load.assert_called_once()
            self.service.select_split.assert_called_once()
            self.mock_validator.validate_structure.assert_called_once()
            self.mock_transformer1.transform.assert_called_once()
            self.mock_transformer2.transform.assert_called_once()
            self.service._validate_keys.assert_called_once()
            self.mock_sampler.sample.assert_called_once()
            self.service._prep_data.assert_called_once()

            # Verify parameters for _prep_data
            self.service._prep_data.assert_called_once_with(
                dataset_info=self.dataset_info,
                sampled_data="sampled_dataset",
                prepper=self.mock_prepper,
            )

        finally:
            # Restore original methods
            self.service.select_split = original_select_split
            self.service._validate_keys = original_validate_keys
            self.service._prep_data = original_prep_data


if __name__ == "__main__":
    unittest.main()
