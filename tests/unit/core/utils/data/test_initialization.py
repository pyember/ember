"""Unit tests for the dataset initialization module.

This module tests the dataset initialization functionality in both the legacy
and unified registry systems.
"""

import unittest
from unittest import mock

from ember.core.utils.data.registry import (
    initialize_registry,
    UNIFIED_REGISTRY,
    UnifiedDatasetRegistry,
)
from ember.core.utils.data.base.models import DatasetInfo, TaskType
from ember.core.utils.data.datasets_registry.truthful_qa import TruthfulQAPrepper
from ember.core.utils.data.datasets_registry.mmlu import MMLUPrepper
from ember.core.utils.data.datasets_registry.commonsense_qa import CommonsenseQAPrepper
from ember.core.utils.data.datasets_registry.halueval import HaluEvalPrepper
from ember.core.utils.data.datasets_registry.short_answer import ShortAnswerPrepper


class TestDatasetInitialization(unittest.TestCase):
    """Test cases for the dataset initialization functions."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Save the original functions
        self.original_initialize_registry = initialize_registry

    def tearDown(self) -> None:
        """Tear down test fixtures."""
        # Restore original function
        globals()["initialize_registry"] = self.original_initialize_registry

    def test_initialize_registry(self) -> None:
        """Test that initialize_registry registers core datasets."""
        # For this test, we'll create a simple mock and verify it gets called
        with mock.patch(
            "ember.core.utils.data.registry.UNIFIED_REGISTRY.register_metadata"
        ) as mock_register:
            # Since initialize_registry is a complex function with dependencies,
            # we'll keep our test simple and focused
            initialize_registry()

            # Verify the register_metadata was called
            self.assertEqual(5, mock_register.call_count)

            # Get the dataset names that were registered
            registered_datasets = {
                call[1]["name"] for call in mock_register.call_args_list
            }
            expected_datasets = {
                "truthful_qa",
                "mmlu",
                "commonsense_qa",
                "halueval",
                "my_shortanswer_ds",
            }

            # Verify all expected datasets were registered
            self.assertEqual(expected_datasets, registered_datasets)

    def test_initialize_registry_with_real_implementation(self) -> None:
        """Test that the actual initialize_registry implementation works correctly."""
        # We'll mock the function calls but preserve their implementation
        with mock.patch(
            "ember.core.utils.data.registry.UNIFIED_REGISTRY.register_metadata"
        ) as mock_register:
            # Call the real initialize_registry
            initialize_registry()

            # Verify it registered the expected datasets
            self.assertEqual(5, mock_register.call_count)

            # Get the dataset names that were registered
            registered_datasets = {
                call[1]["name"] for call in mock_register.call_args_list
            }
            expected_datasets = {
                "truthful_qa",
                "mmlu",
                "commonsense_qa",
                "halueval",
                "my_shortanswer_ds",
            }

            # Verify all expected datasets were registered
            self.assertEqual(expected_datasets, registered_datasets)

            # Create a map of dataset name to prepper class
            prepper_class_map = {
                call[1]["name"]: call[1]["prepper_class"]
                for call in mock_register.call_args_list
            }

            # Verify prepper classes by name instead of direct equality due to import remapping
            self.assertEqual(TruthfulQAPrepper.__name__, prepper_class_map["truthful_qa"].__name__)
            self.assertEqual(MMLUPrepper.__name__, prepper_class_map["mmlu"].__name__)
            self.assertEqual(CommonsenseQAPrepper.__name__, prepper_class_map["commonsense_qa"].__name__)
            self.assertEqual(HaluEvalPrepper.__name__, prepper_class_map["halueval"].__name__)
            self.assertEqual(ShortAnswerPrepper.__name__, prepper_class_map["my_shortanswer_ds"].__name__)

    def test_legacy_initialization_compatibility(self) -> None:
        """Test that the legacy initialization function works with the unified registry."""
        # Import legacy initialization functions
        from ember.core.utils.data.initialization import initialize_dataset_registry

        # Mock the legacy function's dependencies
        with mock.patch(
            "ember.core.utils.data.metadata_registry.UNIFIED_REGISTRY.register_metadata"
        ) as mock_register:
            # Create mock registry and loader factory
            mock_metadata_registry = mock.MagicMock()
            mock_loader_factory = mock.MagicMock()

            # Add required attribute for test to pass
            mock_metadata_registry.register = mock.MagicMock()

            # Call the legacy initialization function
            initialize_dataset_registry(
                metadata_registry=mock_metadata_registry,
                loader_factory=mock_loader_factory,
            )

            # Verify the correct number of calls were made
            self.assertGreaterEqual(mock_metadata_registry.register.call_count, 4)
            self.assertGreaterEqual(mock_loader_factory.register.call_count, 5)

    def test_registry_integration(self) -> None:
        """Test integration between registry components."""
        # Get a reference to the global registry
        from ember.core.utils.data.registry import UNIFIED_REGISTRY
        from ember.core.utils.data.base.models import TaskType

        # Create a clean test registry
        test_registry = UnifiedDatasetRegistry()

        # Mock for the prepper class (since we don't need real functionality)
        prepper_mock = mock.MagicMock()

        # Register a test dataset
        test_registry.register_metadata(
            name="test_dataset",
            description="Test dataset",
            source="test",
            task_type=TaskType.MULTIPLE_CHOICE,
            prepper_class=mock.MagicMock(),
        )

        # Verify we can get the dataset
        dataset = test_registry.get(name="test_dataset")
        self.assertIsNotNone(dataset)
        self.assertEqual("test_dataset", dataset.name if dataset else None)

        # Verify the info is accessible
        info = test_registry.get_info(name="test_dataset")
        self.assertIsNotNone(info)
        self.assertEqual("Test dataset", info.description if info else None)


if __name__ == "__main__":
    unittest.main()
