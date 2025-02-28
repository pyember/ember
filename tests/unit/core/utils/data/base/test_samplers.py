"""Unit tests for dataset samplers."""

import unittest
from typing import Dict, List, Any, Optional
from unittest import mock

from datasets import Dataset
from src.ember.core.utils.data.base.samplers import IDatasetSampler, DatasetSampler


class TestIDatasetSampler(unittest.TestCase):
    """Test cases for the IDatasetSampler interface."""

    def test_interface_enforcement(self) -> None:
        """IDatasetSampler should require implementation of the sample method."""
        # Attempt to instantiate the abstract base class directly
        with self.assertRaises(TypeError):
            IDatasetSampler()  # type: ignore

        # Create a subclass that doesn't implement sample
        class IncompleteSampler(IDatasetSampler):
            pass

        # Attempt to instantiate the incomplete subclass
        with self.assertRaises(TypeError):
            IncompleteSampler()  # type: ignore

        # Create a proper implementation
        class CompleteSampler(IDatasetSampler):
            def sample(self, *, data: Any, num_samples: Optional[int]) -> Any:
                return data

        # Should instantiate without error
        sampler = CompleteSampler()
        self.assertIsInstance(sampler, IDatasetSampler)


class TestDatasetSampler(unittest.TestCase):
    """Test cases for the DatasetSampler class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.sampler = DatasetSampler()

        # Create test data
        self.list_data = [{"id": i, "value": f"test{i}"} for i in range(10)]

        # Create a mock Dataset
        self.mock_dataset = mock.MagicMock(spec=Dataset)
        self.mock_dataset.__len__.return_value = 10
        self.mock_dataset.select.return_value = "selected_dataset"

    def test_sample_with_none_num_samples(self) -> None:
        """sample() should return original data when num_samples is None."""
        # Arrange & Act
        result_list = self.sampler.sample(data=self.list_data, num_samples=None)
        result_dataset = self.sampler.sample(data=self.mock_dataset, num_samples=None)

        # Assert
        self.assertIs(self.list_data, result_list)
        self.assertIs(self.mock_dataset, result_dataset)

    def test_sample_with_zero_num_samples(self) -> None:
        """sample() should return original data when num_samples is 0."""
        # Arrange & Act
        result_list = self.sampler.sample(data=self.list_data, num_samples=0)
        result_dataset = self.sampler.sample(data=self.mock_dataset, num_samples=0)

        # Assert
        self.assertIs(self.list_data, result_list)
        self.assertIs(self.mock_dataset, result_dataset)

    def test_sample_with_negative_num_samples(self) -> None:
        """sample() should return original data when num_samples is negative."""
        # Arrange & Act
        result_list = self.sampler.sample(data=self.list_data, num_samples=-5)
        result_dataset = self.sampler.sample(data=self.mock_dataset, num_samples=-5)

        # Assert
        self.assertIs(self.list_data, result_list)
        self.assertIs(self.mock_dataset, result_dataset)

    def test_sample_list_with_valid_num_samples(self) -> None:
        """sample() should return a slice of the list when num_samples is valid."""
        # Arrange & Act
        num_samples = 5
        result = self.sampler.sample(data=self.list_data, num_samples=num_samples)

        # Assert
        self.assertEqual(num_samples, len(result))
        self.assertEqual(self.list_data[:num_samples], result)

    def test_sample_list_with_excessive_num_samples(self) -> None:
        """sample() should return the entire list when num_samples exceeds list length."""
        # Arrange & Act
        result = self.sampler.sample(data=self.list_data, num_samples=20)

        # Assert
        self.assertEqual(self.list_data, result)

    def test_sample_dataset_with_valid_num_samples(self) -> None:
        """sample() should call dataset.select() with appropriate indices."""
        # Arrange & Act
        num_samples = 5
        result = self.sampler.sample(data=self.mock_dataset, num_samples=num_samples)

        # Assert
        self.assertEqual("selected_dataset", result)
        self.mock_dataset.select.assert_called_once()
        # Check that the indices argument contains integers from 0 to num_samples-1
        call_args = self.mock_dataset.select.call_args[0][0]
        self.assertEqual(list(range(num_samples)), call_args)

    def test_sample_dataset_with_excessive_num_samples(self) -> None:
        """sample() should use dataset length when num_samples exceeds it."""
        # Arrange & Act
        result = self.sampler.sample(data=self.mock_dataset, num_samples=20)

        # Assert
        self.assertEqual("selected_dataset", result)
        self.mock_dataset.select.assert_called_once()
        # Check that the indices argument contains integers from 0 to dataset length - 1
        call_args = self.mock_dataset.select.call_args[0][0]
        self.assertEqual(list(range(10)), call_args)

    def test_sample_with_named_parameters(self) -> None:
        """sample() should require named parameters."""
        # This test verifies that the method enforces the use of named parameters
        # Arrange & Act & Assert
        with self.assertRaises(TypeError):
            self.sampler.sample(self.list_data, 5)  # type: ignore


if __name__ == "__main__":
    unittest.main()
