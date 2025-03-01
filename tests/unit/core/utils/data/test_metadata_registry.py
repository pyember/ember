"""Unit tests for dataset metadata registry."""

import unittest
from typing import Dict, List, Any, Tuple
from unittest import mock

from ember.core.utils.data.base.models import DatasetInfo, TaskType
from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.metadata_registry import (
    DatasetRegistry,
    register_dataset,
    DatasetRegistryManager,
    DatasetMetadataRegistry,
)


class MockPrepper(IDatasetPrepper):
    """Mock implementation of IDatasetPrepper for testing."""

    def get_required_keys(self) -> List[str]:
        """Return a list of required keys.

        Returns:
            List[str]: A list containing 'test'
        """
        return ["test"]

    def create_dataset_entries(self, *, item: Dict[str, Any]) -> List[Any]:
        """Create mock dataset entries.

        Args:
            item (Dict[str, Any]): The item to process.

        Returns:
            List[Any]: A list containing a mock entry.
        """
        return ["mock_entry"]


class TestDatasetRegistry(unittest.TestCase):
    """Test cases for the DatasetRegistry class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Clear the registry before each test
        DatasetRegistry.clear_registry()

        # Create test data
        self.dataset_info = DatasetInfo(
            name="test_dataset",
            description="Test dataset",
            source="test_source",
            task_type=TaskType.MULTIPLE_CHOICE,
        )
        self.prepper = MockPrepper()

    def tearDown(self) -> None:
        """Tear down test fixtures."""
        # Clear registry after each test
        DatasetRegistry.clear_registry()

    def test_register_and_get(self) -> None:
        """register() and get() should store and retrieve datasets correctly."""
        # Arrange & Act
        DatasetRegistry.register(dataset_info=self.dataset_info, prepper=self.prepper)
        result = DatasetRegistry.get(name=self.dataset_info.name)

        # Assert
        self.assertIsNotNone(result)
        info, prepper = result  # type: Tuple[DatasetInfo, IDatasetPrepper]
        self.assertEqual(self.dataset_info, info)
        self.assertEqual(self.prepper, prepper)

    def test_get_nonexistent(self) -> None:
        """get() should return None for nonexistent datasets."""
        # Arrange & Act
        result = DatasetRegistry.get(name="nonexistent")

        # Assert
        self.assertIsNone(result)

    def test_list_datasets(self) -> None:
        """list_datasets() should return all registered dataset names."""
        # Arrange
        datasets = [
            (
                "dataset1",
                DatasetInfo(
                    name="dataset1",
                    description="First test dataset",
                    source="source1",
                    task_type=TaskType.MULTIPLE_CHOICE,
                ),
            ),
            (
                "dataset2",
                DatasetInfo(
                    name="dataset2",
                    description="Second test dataset",
                    source="source2",
                    task_type=TaskType.SHORT_ANSWER,
                ),
            ),
        ]

        # Register datasets
        for name, info in datasets:
            DatasetRegistry.register(dataset_info=info, prepper=self.prepper)

        # Act
        result = DatasetRegistry.list_datasets()

        # Assert
        self.assertEqual(len(datasets), len(result))
        for name, _ in datasets:
            self.assertIn(name, result)

    def test_clear_registry(self) -> None:
        """clear_registry() should remove all registered datasets."""
        # Arrange
        DatasetRegistry.register(dataset_info=self.dataset_info, prepper=self.prepper)

        # Act
        DatasetRegistry.clear_registry()
        result = DatasetRegistry.get(name=self.dataset_info.name)

        # Assert
        self.assertIsNone(result)
        self.assertEqual([], DatasetRegistry.list_datasets())


class TestRegisterDatasetFunction(unittest.TestCase):
    """Test cases for the register_dataset function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Clear the registry before each test
        DatasetRegistry.clear_registry()

        # Mock the prepper class
        self.mock_prepper_cls = mock.MagicMock(spec=type)
        self.mock_prepper = mock.MagicMock(spec=IDatasetPrepper)
        self.mock_prepper_cls.return_value = self.mock_prepper

    def tearDown(self) -> None:
        """Tear down test fixtures."""
        # Clear registry after each test
        DatasetRegistry.clear_registry()

    def test_register_dataset(self) -> None:
        """register_dataset() should create a DatasetInfo and register it with a prepper instance."""
        # Arrange
        name = "test_dataset"
        description = "Test dataset description"
        source = "test_source"
        task_type = TaskType.MULTIPLE_CHOICE

        # Act
        register_dataset(
            name=name,
            description=description,
            source=source,
            task_type=task_type,
            prepper_class=self.mock_prepper_cls,
        )

        # Assert
        result = DatasetRegistry.get(name=name)
        self.assertIsNotNone(result)

        info, prepper = result  # type: Tuple[DatasetInfo, IDatasetPrepper]
        self.assertEqual(name, info.name)
        self.assertEqual(description, info.description)
        self.assertEqual(source, info.source)
        self.assertEqual(task_type, info.task_type)
        self.assertEqual(self.mock_prepper, prepper)

        # Verify prepper class was instantiated
        self.mock_prepper_cls.assert_called_once()


class TestDatasetRegistryManager(unittest.TestCase):
    """Test cases for the DatasetRegistryManager class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.manager = DatasetRegistryManager()

        # Create test data
        self.dataset_info = DatasetInfo(
            name="test_dataset",
            description="Test dataset",
            source="test_source",
            task_type=TaskType.MULTIPLE_CHOICE,
        )

        # Mock prepper class
        self.mock_prepper_cls = mock.MagicMock(spec=type)
        self.mock_prepper = mock.MagicMock(spec=IDatasetPrepper)
        self.mock_prepper_cls.return_value = self.mock_prepper

    def test_register_dataset(self) -> None:
        """register_dataset() should store dataset info and prepper instance."""
        # Arrange & Act
        self.manager.register_dataset(
            info=self.dataset_info, prepper_class=self.mock_prepper_cls
        )

        # Assert
        self.mock_prepper_cls.assert_called_once()

        # Verify dataset info retrieval
        retrieved_info = self.manager.get_dataset_info(name=self.dataset_info.name)
        self.assertEqual(self.dataset_info, retrieved_info)

        # Verify prepper class retrieval
        retrieved_prepper_cls = self.manager.get_prepper_class(
            name=self.dataset_info.name
        )
        self.assertEqual(type(self.mock_prepper), retrieved_prepper_cls)

    def test_get_dataset_info_nonexistent(self) -> None:
        """get_dataset_info() should return None for nonexistent datasets."""
        # Arrange & Act
        result = self.manager.get_dataset_info(name="nonexistent")

        # Assert
        self.assertIsNone(result)

    def test_get_prepper_class_nonexistent(self) -> None:
        """get_prepper_class() should return None for nonexistent datasets."""
        # Arrange & Act
        result = self.manager.get_prepper_class(name="nonexistent")

        # Assert
        self.assertIsNone(result)

    def test_clear_all(self) -> None:
        """clear_all() should remove all registered dataset information."""
        # Arrange
        self.manager.register_dataset(
            info=self.dataset_info, prepper_class=self.mock_prepper_cls
        )

        # Act
        self.manager.clear_all()

        # Assert
        self.assertIsNone(self.manager.get_dataset_info(name=self.dataset_info.name))
        self.assertIsNone(self.manager.get_prepper_class(name=self.dataset_info.name))


class TestDatasetMetadataRegistry(unittest.TestCase):
    """Test cases for the DatasetMetadataRegistry class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.registry = DatasetMetadataRegistry()

        # Create test dataset info
        self.dataset_info = DatasetInfo(
            name="test_dataset",
            description="Test dataset",
            source="test_source",
            task_type=TaskType.MULTIPLE_CHOICE,
        )

    def test_register_and_get(self) -> None:
        """register() and get() should store and retrieve dataset metadata correctly."""
        # Arrange & Act
        self.registry.register(dataset_info=self.dataset_info)
        result = self.registry.get(name=self.dataset_info.name)

        # Assert
        self.assertEqual(self.dataset_info, result)

    def test_get_nonexistent(self) -> None:
        """get() should return None for nonexistent datasets."""
        # Arrange & Act
        result = self.registry.get(name="nonexistent")

        # Assert
        self.assertIsNone(result)

    def test_list_datasets(self) -> None:
        """list_datasets() should return all registered dataset names."""
        # Arrange
        datasets = [
            DatasetInfo(
                name="dataset1",
                description="First test dataset",
                source="source1",
                task_type=TaskType.MULTIPLE_CHOICE,
            ),
            DatasetInfo(
                name="dataset2",
                description="Second test dataset",
                source="source2",
                task_type=TaskType.SHORT_ANSWER,
            ),
        ]

        # Register datasets
        for info in datasets:
            self.registry.register(dataset_info=info)

        # Act
        result = self.registry.list_datasets()

        # Assert
        self.assertEqual(len(datasets), len(result))
        for info in datasets:
            self.assertIn(info.name, result)


if __name__ == "__main__":
    unittest.main()
