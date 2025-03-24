"""Unit tests for the dataset registry compatibility.

This module tests the compatibility between the legacy and unified registry systems,
especially focusing on the fixes we made to ensure backward compatibility.
"""

import unittest
from unittest import mock

from ember.core.utils.data.base.models import DatasetInfo, TaskType
from ember.core.utils.data.initialization import initialize_dataset_registry
from ember.core.utils.data.registry import UNIFIED_REGISTRY, UnifiedDatasetRegistry


class TestRegistryCompatibility(unittest.TestCase):
    """Test cases for the dataset registry compatibility."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Save the original registry
        self.original_registry = UNIFIED_REGISTRY

        # Create a clean test registry
        self.test_registry = UnifiedDatasetRegistry()

    def tearDown(self) -> None:
        """Tear down test fixtures."""
        # Restore original registry
        globals()["UNIFIED_REGISTRY"] = self.original_registry

    def test_initialization_with_register_new(self) -> None:
        """Test initialization using register_new instead of register."""
        # Create a custom DatasetInfo
        test_info = DatasetInfo(
            name="test_dataset",
            description="Test dataset",
            source="test_source",
            task_type=TaskType.MULTIPLE_CHOICE,
        )

        # Create mocks
        mock_loader_factory = mock.MagicMock()

        # Register using register_new method
        self.test_registry.register_new(name=test_info.name, info=test_info)

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
        """Test that our registry accessor compatibility works."""
        # Create a normal UnifiedDatasetRegistry
        registry = UnifiedDatasetRegistry()

        # Create a test DatasetInfo
        test_info = DatasetInfo(
            name="test_dataset",
            description="Test dataset",
            source="test_source",
            task_type=TaskType.MULTIPLE_CHOICE,
        )

        # Register using register_new
        registry.register_new(name=test_info.name, info=test_info)

        # Importing the function that has our compatibility code
        from ember.core.utils.data import initialize_dataset_registry

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
