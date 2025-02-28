"""Unit tests for the data module's high-level API."""

import unittest
from typing import Any, Dict, List, Optional, Type, Union
from unittest import mock

from src.ember.core.utils.data import (
    load_dataset_entries,
    DatasetMetadataRegistry,
    DatasetLoaderFactory,
    DatasetInfo,
    TaskType,
    BaseDatasetConfig,
    HuggingFaceDatasetLoader,
    DatasetValidator,
    DatasetSampler,
    DatasetService,
    IDatasetPrepper,
)
from src.ember.core.utils.data.initialization import initialize_dataset_registry


class TestLoadDatasetEntries(unittest.TestCase):
    """Test cases for the load_dataset_entries function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create patches for the dependencies
        self.metadata_registry_patcher = mock.patch(
            "src.ember.core.utils.data.DatasetMetadataRegistry"
        )
        self.mock_metadata_registry_cls = self.metadata_registry_patcher.start()
        self.mock_metadata_registry = self.mock_metadata_registry_cls.return_value

        self.loader_factory_patcher = mock.patch(
            "src.ember.core.utils.data.DatasetLoaderFactory"
        )
        self.mock_loader_factory_cls = self.loader_factory_patcher.start()
        self.mock_loader_factory = self.mock_loader_factory_cls.return_value

        self.initialize_registry_patcher = mock.patch(
            "src.ember.core.utils.data.initialize_dataset_registry"
        )
        self.mock_initialize_registry = self.initialize_registry_patcher.start()

        self.hf_loader_patcher = mock.patch(
            "src.ember.core.utils.data.HuggingFaceDatasetLoader"
        )
        self.mock_hf_loader_cls = self.hf_loader_patcher.start()
        self.mock_hf_loader = self.mock_hf_loader_cls.return_value

        self.validator_patcher = mock.patch(
            "src.ember.core.utils.data.DatasetValidator"
        )
        self.mock_validator_cls = self.validator_patcher.start()
        self.mock_validator = self.mock_validator_cls.return_value

        self.sampler_patcher = mock.patch("src.ember.core.utils.data.DatasetSampler")
        self.mock_sampler_cls = self.sampler_patcher.start()
        self.mock_sampler = self.mock_sampler_cls.return_value

        self.service_patcher = mock.patch("src.ember.core.utils.data.DatasetService")
        self.mock_service_cls = self.service_patcher.start()
        self.mock_service = self.mock_service_cls.return_value

        # Configure mock metadata registry
        self.mock_dataset_info = DatasetInfo(
            name="test_dataset",
            description="Test dataset",
            source="test_source",
            task_type=TaskType.MULTIPLE_CHOICE,
        )
        self.mock_metadata_registry.get.return_value = self.mock_dataset_info

        # Configure mock loader factory
        self.mock_prepper_cls = mock.MagicMock(spec=type)
        self.mock_prepper = mock.MagicMock(spec=IDatasetPrepper)
        self.mock_prepper_cls.return_value = self.mock_prepper
        self.mock_loader_factory.get_prepper_class.return_value = self.mock_prepper_cls

        # Configure mock service
        self.mock_entries = [mock.MagicMock(), mock.MagicMock()]
        self.mock_service.load_and_prepare.return_value = self.mock_entries

    def tearDown(self) -> None:
        """Tear down test fixtures."""
        self.metadata_registry_patcher.stop()
        self.loader_factory_patcher.stop()
        self.initialize_registry_patcher.stop()
        self.hf_loader_patcher.stop()
        self.validator_patcher.stop()
        self.sampler_patcher.stop()
        self.service_patcher.stop()

    def test_load_dataset_entries_success(self) -> None:
        """load_dataset_entries() should orchestrate the full dataset loading pipeline."""
        # Arrange
        dataset_name = "test_dataset"
        config = mock.MagicMock(spec=BaseDatasetConfig)
        num_samples = 5

        # Act
        result = load_dataset_entries(
            dataset_name=dataset_name, config=config, num_samples=num_samples
        )

        # Assert
        self.assertEqual(self.mock_entries, result)

        # Verify initialization flow
        self.mock_metadata_registry_cls.assert_called_once()
        self.mock_loader_factory_cls.assert_called_once()
        self.mock_initialize_registry.assert_called_once_with(
            metadata_registry=self.mock_metadata_registry,
            loader_factory=self.mock_loader_factory,
        )
        self.mock_loader_factory.discover_and_register_plugins.assert_called_once()

        # Verify metadata and prepper retrieval
        self.mock_metadata_registry.get.assert_called_once_with(dataset_name)
        self.mock_loader_factory.get_prepper_class.assert_called_once_with(
            dataset_name=dataset_name
        )

        # Verify prepper instantiation
        self.mock_prepper_cls.assert_called_once_with(config=config)

        # Verify service creation and usage
        self.mock_hf_loader_cls.assert_called_once()
        self.mock_validator_cls.assert_called_once()
        self.mock_sampler_cls.assert_called_once()
        self.mock_service_cls.assert_called_once_with(
            loader=self.mock_hf_loader,
            validator=self.mock_validator,
            sampler=self.mock_sampler,
        )
        self.mock_service.load_and_prepare.assert_called_once_with(
            dataset_info=self.mock_dataset_info,
            prepper=self.mock_prepper,
            config=config,
            num_samples=num_samples,
        )

    def test_load_dataset_entries_with_string_config(self) -> None:
        """load_dataset_entries() should handle string config parameter."""
        # Arrange
        dataset_name = "test_dataset"
        config = "test_config"

        # Act
        result = load_dataset_entries(dataset_name=dataset_name, config=config)

        # Assert
        self.assertEqual(self.mock_entries, result)
        self.mock_service.load_and_prepare.assert_called_once_with(
            dataset_info=self.mock_dataset_info,
            prepper=self.mock_prepper,
            config=config,
            num_samples=None,
        )

    def test_load_dataset_entries_dataset_not_found(self) -> None:
        """load_dataset_entries() should raise ValueError when dataset is not found."""
        # Arrange
        dataset_name = "nonexistent_dataset"
        self.mock_metadata_registry.get.return_value = None

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            load_dataset_entries(dataset_name=dataset_name)

        self.assertIn(dataset_name, str(context.exception))

    def test_load_dataset_entries_prepper_not_found(self) -> None:
        """load_dataset_entries() should raise ValueError when prepper is not found."""
        # Arrange
        dataset_name = "test_dataset"
        self.mock_loader_factory.get_prepper_class.return_value = None

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            load_dataset_entries(dataset_name=dataset_name)

        self.assertIn(dataset_name, str(context.exception))

    def test_load_dataset_entries_with_named_params(self) -> None:
        """load_dataset_entries() should require named parameters."""
        # Arrange & Act & Assert
        with self.assertRaises(TypeError):
            load_dataset_entries("test_dataset")  # type: ignore


if __name__ == "__main__":
    unittest.main()
