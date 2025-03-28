"""Unit tests for the dataset registry compatibility.

This module tests the compatibility between the legacy and unified registry systems,
especially focusing on the fixes we made to ensure backward compatibility.
"""

import unittest
from unittest import mock

from ember.core.utils.data.base.models import DatasetInfo, TaskType
from ember.core.utils.data.initialization import initialize_dataset_registry
from ember.core.utils.data.registry import DATASET_REGISTRY, DatasetRegistry


class TestRegistryCompatibility(unittest.TestCase):
    """Test cases for the dataset registry compatibility."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Save the original registry
        self.original_registry = DATASET_REGISTRY

        # Create a clean test registry
        self.test_registry = DatasetRegistry()

    def tearDown(self) -> None:
        """Tear down test fixtures."""
        # Restore original registry
        globals()["DATASET_REGISTRY"] = self.original_registry

    def test_initialization_with_register(self) -> None:
        """Test initialization using register method."""
        # Create a custom DatasetInfo
        test_info = DatasetInfo(
            name="test_dataset",
            description="Test dataset",
            source="test_source",
            task_type=TaskType.MULTIPLE_CHOICE,
        )

        # Create mocks
        mock_loader_factory = mock.MagicMock()

        # Register using register method (the new standard API)
        self.test_registry.register(name=test_info.name, info=test_info)

        # Call the initialization function with our test registry
        initialize_dataset_registry(
            metadata_registry=self.test_registry,
            loader_factory=mock_loader_factory,
        )

        # Verify the registry can find our test dataset
        retrieved_info = self.test_registry.get_info(name="test_dataset")
        self.assertIsNotNone(retrieved_info)
        self.assertEqual(
            "Test dataset", retrieved_info.description if retrieved_info else None
        )

        # Verify standard datasets were also registered
        self.assertIsNotNone(self.test_registry.get_info(name="mmlu"))
        self.assertIsNotNone(self.test_registry.get_info(name="truthful_qa"))

    def test_registry_accessor_compatibility(self) -> None:
        """Test that our registry accessor patterns work."""
        # Create a normal DatasetRegistry
        registry = DatasetRegistry()

        # Create a test DatasetInfo
        test_info = DatasetInfo(
            name="test_dataset",
            description="Test dataset",
            source="test_source",
            task_type=TaskType.MULTIPLE_CHOICE,
        )

        # Register using standard register method
        registry.register(name=test_info.name, info=test_info)

        # Create helper class to test different access patterns
        class RegistryAccessor:
            def __init__(self, registry):
                self.registry = registry

            def test_get_info_method(self):
                """Test get_info method."""
                info = self.registry.get_info(name="test_dataset")
                return info.description if info else None

            def test_get_with_named_params(self):
                """Test get method with named parameters."""
                dataset = self.registry.get(name="test_dataset")
                return dataset.info.description if dataset and dataset.info else None

        # Test access patterns
        accessor = RegistryAccessor(registry)

        # Check all access patterns
        self.assertEqual("Test dataset", accessor.test_get_info_method())
        self.assertEqual("Test dataset", accessor.test_get_with_named_params())


if __name__ == "__main__":
    unittest.main()
