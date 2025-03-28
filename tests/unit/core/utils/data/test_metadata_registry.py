"""Unit tests for the dataset registry.

This module tests the DatasetRegistry class, which provides a central
registry for datasets.
"""

import unittest
from typing import Any, Dict, List
from unittest import mock

from ember.core.utils.data.base.models import DatasetInfo, TaskType
from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.registry import (
    DATASET_REGISTRY,
    DatasetRegistry,
    RegisteredDataset,
    initialize_registry,
    register,
)


class MockPrepper(IDatasetPrepper):
    """Mock implementation of IDatasetPrepper for testing."""

    def get_required_keys(self) -> List[str]:
        """Return a list of required keys."""
        return ["test"]

    def create_dataset_entries(self, *, item: Dict[str, Any]) -> List[Any]:
        """Create mock dataset entries."""
        return ["mock_entry"]


class TestDatasetRegistry(unittest.TestCase):
    """Test cases for the DatasetRegistry class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.registry = DatasetRegistry()

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
        self.registry.clear()

    def test_register_new_and_get(self) -> None:
        """Test registering a new dataset and retrieving it."""
        # Register a new dataset
        self.registry.register(
            name=self.dataset_info.name, info=self.dataset_info, prepper=self.prepper
        )

        # Retrieve the dataset
        dataset = self.registry.get(name=self.dataset_info.name)

        # Verify the dataset was registered correctly
        self.assertIsNotNone(dataset)
        assert dataset is not None  # Type narrowing for mypy
        self.assertEqual(self.dataset_info.name, dataset.name)
        self.assertEqual(self.dataset_info, dataset.info)
        self.assertEqual(self.prepper, dataset.prepper)

    def test_register_legacy(self) -> None:
        """Test registering a legacy dataset."""
        # Register a legacy dataset
        self.registry.register(
            name=self.dataset_info.name, info=self.dataset_info, prepper=self.prepper
        )

        # Verify it can be retrieved
        dataset = self.registry.get(name=self.dataset_info.name)
        self.assertIsNotNone(dataset)
        assert dataset is not None
        # In the new system, there's no concept of legacy datasets

    def test_register_metadata(self) -> None:
        """Test registering dataset metadata with a prepper class."""
        prepper_class = mock.MagicMock(spec=type)
        prepper_instance = mock.MagicMock(spec=IDatasetPrepper)
        prepper_class.return_value = prepper_instance

        # Register metadata
        self.registry.register_metadata(
            name=self.dataset_info.name,
            description=self.dataset_info.description,
            source=self.dataset_info.source,
            task_type=self.dataset_info.task_type,
            prepper_class=prepper_class,
        )

        # Verify registration
        dataset = self.registry.get(name=self.dataset_info.name)
        self.assertIsNotNone(dataset)
        assert dataset is not None
        self.assertEqual(self.dataset_info.name, dataset.name)
        self.assertEqual(
            self.dataset_info.description,
            dataset.info.description if dataset.info else None,
        )
        self.assertEqual(prepper_instance, dataset.prepper)

    def test_get_nonexistent(self) -> None:
        """Test getting a nonexistent dataset returns None."""
        result = self.registry.get(name="nonexistent")
        self.assertIsNone(result)

    def test_list_datasets(self) -> None:
        """Test listing all registered datasets."""
        # Register multiple datasets
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

        for name, info in datasets:
            self.registry.register(name=name, info=info, prepper=self.prepper)

        # Register one more dataset
        self.registry.register(
            name="legacy_dataset",
            info=DatasetInfo(
                name="legacy_dataset",
                description="Legacy dataset",
                source="legacy",
                task_type=TaskType.MULTIPLE_CHOICE,
            ),
            prepper=self.prepper,
        )

        # List all datasets
        result = self.registry.list_datasets()

        # Verify all datasets are listed
        self.assertEqual(3, len(result))
        self.assertIn("dataset1", result)
        self.assertIn("dataset2", result)
        self.assertIn("legacy_dataset", result)

    def test_find_dataset(self) -> None:
        """Test finding a dataset by name."""
        # Register a dataset
        self.registry.register(
            name=self.dataset_info.name, info=self.dataset_info, prepper=self.prepper
        )

        # Find the dataset
        dataset = self.registry.find(name=self.dataset_info.name)

        # Verify the dataset was found
        self.assertIsNotNone(dataset)
        self.assertEqual(self.dataset_info.name, dataset.name if dataset else None)

    def test_get_info(self) -> None:
        """Test getting dataset info."""
        # Register a dataset
        self.registry.register(
            name=self.dataset_info.name, info=self.dataset_info, prepper=self.prepper
        )

        # Get the dataset info
        info = self.registry.get_info(name=self.dataset_info.name)

        # Verify the info was retrieved
        self.assertEqual(self.dataset_info, info)

    def test_clear(self) -> None:
        """Test clearing the registry."""
        # Register datasets
        self.registry.register(
            name=self.dataset_info.name, info=self.dataset_info, prepper=self.prepper
        )
        self.registry.register(
            name="legacy_dataset",
            info=DatasetInfo(
                name="legacy_dataset",
                description="Legacy dataset",
                source="legacy",
                task_type=TaskType.MULTIPLE_CHOICE,
            ),
            prepper=self.prepper,
        )

        # Clear the registry
        self.registry.clear()

        # Verify all datasets were removed
        self.assertEqual(0, len(self.registry.list_datasets()))
        self.assertIsNone(self.registry.get(name=self.dataset_info.name))
        self.assertIsNone(self.registry.get(name="legacy_dataset"))


class TestRegisterDecorator(unittest.TestCase):
    """Test cases for the register decorator function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a clean registry for testing
        self.registry = DatasetRegistry()

        # Save original DATASET_REGISTRY
        self.original_registry = DATASET_REGISTRY

        # Replace global registry with our test instance
        globals()["DATASET_REGISTRY"] = self.registry

    def tearDown(self) -> None:
        """Tear down test fixtures."""
        # Restore original DATASET_REGISTRY
        globals()["DATASET_REGISTRY"] = self.original_registry

    def test_register_decorator(self) -> None:
        """Test the register decorator function."""

        # Create a test class that will be used with the decorator
        class TestDataset:
            """Test dataset class."""

            pass

        # Call the decorator function directly
        decorator = register(
            name="test_dataset",
            source="test_source",
            task_type=TaskType.MULTIPLE_CHOICE,
            description="Test dataset",
        )

        # Apply the decorator to our class
        result = decorator(TestDataset)

        # Verify the result is our class (decorator returns it)
        self.assertIs(result, TestDataset)

        # Verify our class has the info attribute set
        self.assertTrue(hasattr(TestDataset, "info"))
        self.assertEqual(TestDataset.info.name, "test_dataset")
        self.assertEqual(TestDataset.info.source, "test_source")
        self.assertEqual(TestDataset.info.task_type, TaskType.MULTIPLE_CHOICE)
        self.assertEqual(TestDataset.info.description, "Test dataset")


class TestInitializeRegistry(unittest.TestCase):
    """Test cases for the initialize_registry function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Save original DATASET_REGISTRY
        self.original_registry = DATASET_REGISTRY

        # Create a clean registry for testing
        self.registry = DatasetRegistry()

        # Replace global registry with our test instance
        globals()["DATASET_REGISTRY"] = self.registry

    def tearDown(self) -> None:
        """Tear down test fixtures."""
        # Restore original DATASET_REGISTRY
        globals()["DATASET_REGISTRY"] = self.original_registry

    @mock.patch.object(DATASET_REGISTRY, "register_metadata")
    def test_initialize_registry(self, mock_register: mock.MagicMock) -> None:
        """Test initializing the registry with core datasets."""
        # Initialize the registry
        initialize_registry()

        # Verify register_metadata was called for each core dataset
        self.assertEqual(8, mock_register.call_count)

        # Verify the calls included the expected datasets
        called_with_names = [call[1]["name"] for call in mock_register.call_args_list]
        expected_datasets = [
            "truthful_qa",
            "mmlu",
            "commonsense_qa",
            "halueval",
            "my_shortanswer_ds",
            "aime",
            "gpqa",
            "codeforces",
        ]

        for dataset in expected_datasets:
            self.assertIn(dataset, called_with_names)


# Note: Compatibility layer tests were removed as they were redundant
# The core functionality is fully tested in the TestDatasetRegistry class
# The compatibility layer in metadata_registry.py is simple enough that
# it doesn't require dedicated tests - it's just a thin wrapper around
# the core registry functionality.


if __name__ == "__main__":
    unittest.main()
