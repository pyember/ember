"""Unit tests for dataset transformers."""

import unittest
from typing import Dict, List, Any, Optional
from unittest import mock

from datasets import Dataset
from src.ember.core.utils.data.base.transformers import (
    IDatasetTransformer,
    NoOpTransformer,
)


class TestIDatasetTransformer(unittest.TestCase):
    """Test cases for the IDatasetTransformer interface."""

    def test_interface_enforcement(self) -> None:
        """IDatasetTransformer should require implementation of the transform method."""
        # Attempt to instantiate the abstract base class directly
        with self.assertRaises(TypeError):
            IDatasetTransformer()  # type: ignore

        # Create a subclass that doesn't implement transform
        class IncompleteTransformer(IDatasetTransformer):
            pass

        # Attempt to instantiate the incomplete subclass
        with self.assertRaises(TypeError):
            IncompleteTransformer()  # type: ignore

        # Create a proper implementation
        class CompleteTransformer(IDatasetTransformer):
            def transform(self, *, data: Any) -> Any:
                return data

        # Should instantiate without error
        transformer = CompleteTransformer()
        self.assertIsInstance(transformer, IDatasetTransformer)


class ConcreteTransformer(IDatasetTransformer):
    """Concrete implementation of IDatasetTransformer for testing purposes."""

    def transform(self, *, data: Any) -> Any:
        """Transform the data by adding a constant field to each record.

        Args:
            data (Any): Input dataset to transform.

        Returns:
            Any: Transformed dataset.
        """
        if isinstance(data, Dataset):
            # For Hugging Face Datasets, we would need to use their API to add a column
            # For testing, just return a modified mock
            result = mock.MagicMock(spec=Dataset)
            result.has_been_transformed = True
            return result
        elif isinstance(data, list):
            # For list data, add a field to each record
            return [dict(item, **{"transformed": True}) for item in data]
        else:
            # For unsupported types, return as is
            return data


class TestNoOpTransformer(unittest.TestCase):
    """Test cases for the NoOpTransformer class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.transformer = NoOpTransformer()

        # Create test data
        self.list_data = [{"id": i, "value": f"test{i}"} for i in range(3)]
        self.mock_dataset = mock.MagicMock(spec=Dataset)

    def test_transform_list_data(self) -> None:
        """transform() should return the original list data unchanged."""
        # Arrange & Act
        result = self.transformer.transform(data=self.list_data)

        # Assert
        self.assertIs(self.list_data, result)

    def test_transform_dataset(self) -> None:
        """transform() should return the original Dataset unchanged."""
        # Arrange & Act
        result = self.transformer.transform(data=self.mock_dataset)

        # Assert
        self.assertIs(self.mock_dataset, result)

    def test_transform_with_named_parameters(self) -> None:
        """transform() should require named parameters."""
        # This test verifies that the method enforces the use of named parameters
        # Arrange & Act & Assert
        with self.assertRaises(TypeError):
            self.transformer.transform(self.list_data)  # type: ignore


class TestConcreteTransformer(unittest.TestCase):
    """Test cases for a concrete implementation of IDatasetTransformer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.transformer = ConcreteTransformer()

        # Create test data
        self.list_data = [{"id": i, "value": f"test{i}"} for i in range(3)]
        self.mock_dataset = mock.MagicMock(spec=Dataset)

    def test_transform_list_data(self) -> None:
        """transform() should add a 'transformed' field to each record in list data."""
        # Arrange & Act
        result = self.transformer.transform(data=self.list_data)

        # Assert
        self.assertEqual(len(self.list_data), len(result))
        for item in result:
            self.assertTrue(item["transformed"])

        # Verify original data is unchanged
        for item in self.list_data:
            self.assertFalse("transformed" in item)

    def test_transform_dataset(self) -> None:
        """transform() should modify the Dataset according to the implementation."""
        # Arrange & Act
        result = self.transformer.transform(data=self.mock_dataset)

        # Assert
        self.assertIsNot(self.mock_dataset, result)
        self.assertTrue(result.has_been_transformed)

    def test_chaining_transformers(self) -> None:
        """Multiple transformers should be able to be chained together."""
        # Arrange
        second_transformer = ConcreteTransformer()

        # Act
        # Apply first transformer
        intermediate_result = self.transformer.transform(data=self.list_data)
        # Apply second transformer
        final_result = second_transformer.transform(data=intermediate_result)

        # Assert
        self.assertEqual(len(self.list_data), len(final_result))
        for item in final_result:
            self.assertTrue(item["transformed"])


if __name__ == "__main__":
    unittest.main()
