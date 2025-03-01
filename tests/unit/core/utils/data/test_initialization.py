"""Unit tests for the dataset initialization module."""

import unittest
from unittest import mock

from ember.core.utils.data.initialization import initialize_dataset_registry
from ember.core.utils.data.metadata_registry import DatasetMetadataRegistry
from ember.core.utils.data.loader_factory import DatasetLoaderFactory

from ember.core.utils.data.datasets_registry.truthful_qa import TruthfulQAPrepper
from ember.core.utils.data.datasets_registry.mmlu import MMLUPrepper
from ember.core.utils.data.datasets_registry.commonsense_qa import (
    CommonsenseQAPrepper,
)
from ember.core.utils.data.datasets_registry.halueval import HaluEvalPrepper
from ember.core.utils.data.datasets_registry.short_answer import ShortAnswerPrepper


class TestInitializeDatasetRegistry(unittest.TestCase):
    """Test cases for the initialize_dataset_registry function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create mock metadata registry
        self.mock_metadata_registry = mock.MagicMock(spec=DatasetMetadataRegistry)

        # Create mock loader factory
        self.mock_loader_factory = mock.MagicMock(spec=DatasetLoaderFactory)

    def test_initialize_dataset_registry(self) -> None:
        """initialize_dataset_registry() should register datasets and their preppers."""
        # Arrange & Act
        initialize_dataset_registry(
            metadata_registry=self.mock_metadata_registry,
            loader_factory=self.mock_loader_factory,
        )

        # Assert
        # Verify metadata registrations
        self.assertGreaterEqual(self.mock_metadata_registry.register.call_count, 4)

        # Verify prepper registrations
        self.assertGreaterEqual(self.mock_loader_factory.register.call_count, 5)

        # Verify specific dataset registrations - extract args from all calls
        metadata_register_args = [
            call[1].get("dataset_info").name
            for call in self.mock_metadata_registry.register.call_args_list
        ]

        # Check that important datasets were registered
        self.assertIn("truthful_qa", metadata_register_args)
        self.assertIn("mmlu", metadata_register_args)
        self.assertIn("commonsense_qa", metadata_register_args)
        self.assertIn("halueval", metadata_register_args)

        # Verify specific prepper registrations - extract args from all calls
        prepper_register_calls = self.mock_loader_factory.register.call_args_list
        registered_datasets = set()
        registered_preppers = set()

        for call in prepper_register_calls:
            kwargs = call[1]
            registered_datasets.add(kwargs.get("dataset_name"))
            registered_preppers.add(kwargs.get("prepper_class"))

        # Check all datasets were registered
        self.assertIn("truthful_qa", registered_datasets)
        self.assertIn("mmlu", registered_datasets)
        self.assertIn("commonsense_qa", registered_datasets)
        self.assertIn("halueval", registered_datasets)
        self.assertIn("my_shortanswer_ds", registered_datasets)

        # Check all preppers were registered
        self.assertIn(TruthfulQAPrepper, registered_preppers)
        self.assertIn(MMLUPrepper, registered_preppers)
        self.assertIn(CommonsenseQAPrepper, registered_preppers)
        self.assertIn(HaluEvalPrepper, registered_preppers)
        self.assertIn(ShortAnswerPrepper, registered_preppers)

    def test_initialize_dataset_registry_with_named_params(self) -> None:
        """initialize_dataset_registry() should require named parameters."""
        # This test verifies that the method enforces the use of named parameters
        # Arrange & Act & Assert
        with self.assertRaises(TypeError):
            initialize_dataset_registry(self.mock_metadata_registry, self.mock_loader_factory)  # type: ignore


if __name__ == "__main__":
    unittest.main()
