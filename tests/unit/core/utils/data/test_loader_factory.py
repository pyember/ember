"""Unit tests for the DatasetLoaderFactory class."""

import unittest
from typing import Dict, List, Any
from unittest import mock

from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.loader_factory import (
    DatasetLoaderFactory,
    discover_preppers,
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


class TestDiscoverPreppers(unittest.TestCase):
    """Test cases for the discover_preppers function."""

    @mock.patch("ember.core.utils.data.loader_factory.entry_points")
    def test_discover_preppers_success(self, mock_entry_points):
        """discover_preppers() should find and load valid prepper classes."""
        # Arrange
        mock_entry_point1 = mock.Mock()
        mock_entry_point1.name = "dataset1"
        mock_entry_point1.load.return_value = MockPrepper

        mock_entry_point2 = mock.Mock()
        mock_entry_point2.name = "dataset2"
        mock_entry_point2.load.return_value = MockPrepper

        mock_select = mock.Mock()
        mock_select.select.return_value = [mock_entry_point1, mock_entry_point2]
        mock_entry_points.return_value = mock_select

        # Act
        result = discover_preppers(entry_point_group="test.entry_points")

        # Assert
        mock_select.select.assert_called_once_with(group="test.entry_points")
        self.assertEqual(2, len(result))
        self.assertIn("dataset1", result)
        self.assertIn("dataset2", result)
        self.assertEqual(MockPrepper, result["dataset1"])
        self.assertEqual(MockPrepper, result["dataset2"])

    @mock.patch("ember.core.utils.data.loader_factory.entry_points")
    def test_discover_preppers_error_handling(self, mock_entry_points):
        """discover_preppers() should handle errors when loading entry points."""
        # Arrange
        mock_entry_point1 = mock.Mock()
        mock_entry_point1.name = "dataset1"
        mock_entry_point1.load.return_value = MockPrepper

        mock_entry_point2 = mock.Mock()
        mock_entry_point2.name = "dataset2"
        mock_entry_point2.load.side_effect = ImportError("Test error")

        mock_select = mock.Mock()
        mock_select.select.return_value = [mock_entry_point1, mock_entry_point2]
        mock_entry_points.return_value = mock_select

        # Act
        result = discover_preppers(entry_point_group="test.entry_points")

        # Assert
        self.assertEqual(1, len(result))
        self.assertIn("dataset1", result)
        self.assertNotIn("dataset2", result)

    @mock.patch("ember.core.utils.data.loader_factory.entry_points")
    def test_discover_preppers_empty(self, mock_entry_points):
        """discover_preppers() should return an empty dict when no entry points found."""
        # Arrange
        mock_select = mock.Mock()
        mock_select.select.return_value = []
        mock_entry_points.return_value = mock_select

        # Act
        result = discover_preppers(entry_point_group="test.entry_points")

        # Assert
        self.assertEqual({}, result)


class TestDatasetLoaderFactory(unittest.TestCase):
    """Test cases for the DatasetLoaderFactory class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.factory = DatasetLoaderFactory()

        # Create a patcher for discover_preppers
        self.discover_preppers_patcher = mock.patch(
            "ember.core.utils.data.loader_factory.discover_preppers"
        )
        self.mock_discover_preppers = self.discover_preppers_patcher.start()

    def tearDown(self) -> None:
        """Tear down test fixtures."""
        self.discover_preppers_patcher.stop()

    def test_initial_state(self) -> None:
        """DatasetLoaderFactory should initialize with an empty registry."""
        # Arrange & Act
        factory = DatasetLoaderFactory()

        # Assert
        self.assertEqual([], factory.list_registered_preppers())

    def test_register_and_get(self) -> None:
        """register() and get_prepper_class() should store and retrieve prepper classes."""
        # Arrange
        dataset_name = "test_dataset"

        # Act
        self.factory.register(dataset_name=dataset_name, prepper_class=MockPrepper)
        result = self.factory.get_prepper_class(dataset_name=dataset_name)

        # Assert
        self.assertEqual(MockPrepper, result)

    def test_get_nonexistent(self) -> None:
        """get_prepper_class() should return None for nonexistent datasets."""
        # Arrange & Act
        result = self.factory.get_prepper_class(dataset_name="nonexistent")

        # Assert
        self.assertIsNone(result)

    def test_list_registered_preppers(self) -> None:
        """list_registered_preppers() should return all registered dataset names."""
        # Arrange
        datasets = {"dataset1": MockPrepper, "dataset2": MockPrepper}

        # Register datasets
        for name, prepper_cls in datasets.items():
            self.factory.register(dataset_name=name, prepper_class=prepper_cls)

        # Act
        result = self.factory.list_registered_preppers()

        # Assert
        self.assertEqual(len(datasets), len(result))
        for name in datasets:
            self.assertIn(name, result)

    def test_clear(self) -> None:
        """clear() should remove all registered prepper classes."""
        # Arrange
        self.factory.register(dataset_name="test_dataset", prepper_class=MockPrepper)

        # Act
        self.factory.clear()
        result = self.factory.get_prepper_class(dataset_name="test_dataset")

        # Assert
        self.assertIsNone(result)
        self.assertEqual([], self.factory.list_registered_preppers())

    def test_discover_and_register_plugins(self) -> None:
        """discover_and_register_plugins() should find and register prepper plugins."""
        # Arrange
        discovered_preppers = {"dataset1": MockPrepper, "dataset2": MockPrepper}
        self.mock_discover_preppers.return_value = discovered_preppers

        # Act
        self.factory.discover_and_register_plugins()

        # Assert
        self.mock_discover_preppers.assert_called_once_with(
            entry_point_group="ember.dataset_preppers"
        )

        # Verify all discovered preppers were registered
        registered_preppers = self.factory.list_registered_preppers()
        self.assertEqual(len(discovered_preppers), len(registered_preppers))
        for name in discovered_preppers:
            self.assertIn(name, registered_preppers)
            self.assertEqual(
                discovered_preppers[name],
                self.factory.get_prepper_class(dataset_name=name),
            )

    def test_register_with_named_parameters(self) -> None:
        """register() should require named parameters."""
        # This test verifies that the method enforces the use of named parameters
        # Arrange & Act & Assert
        with self.assertRaises(TypeError):
            self.factory.register("test_dataset", MockPrepper)  # type: ignore

    def test_get_prepper_class_with_named_parameters(self) -> None:
        """get_prepper_class() should require named parameters."""
        # Arrange & Act & Assert
        with self.assertRaises(TypeError):
            self.factory.get_prepper_class("test_dataset")  # type: ignore


if __name__ == "__main__":
    unittest.main()
