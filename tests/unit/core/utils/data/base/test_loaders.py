"""Unit tests for dataset loader classes."""

import os
import unittest
from unittest import mock
from typing import Optional

from datasets import Dataset, DatasetDict
from urllib.error import HTTPError

from ember.core.utils.data.base.loaders import (
    IDatasetLoader,
    HuggingFaceDatasetLoader,
)


class TestIDatasetLoader(unittest.TestCase):
    """Test cases for the IDatasetLoader interface."""

    def test_interface_enforcement(self) -> None:
        """IDatasetLoader should require implementation of the load method."""
        # Attempt to instantiate the abstract base class directly
        with self.assertRaises(TypeError):
            IDatasetLoader()  # type: ignore

        # Create a subclass that doesn't implement load
        class IncompleteLoader(IDatasetLoader):
            pass

        # Attempt to instantiate the incomplete subclass
        with self.assertRaises(TypeError):
            IncompleteLoader()  # type: ignore

        # Create a proper implementation
        class CompleteLoader(IDatasetLoader):
            def load(
                self, *, dataset_name: str, config: Optional[str] = None
            ) -> Dataset:
                return Dataset.from_dict({"data": [1, 2, 3]})

        # Should instantiate without error
        loader = CompleteLoader()
        self.assertIsInstance(loader, IDatasetLoader)


class TestHuggingFaceDatasetLoader(unittest.TestCase):
    """Test cases for the HuggingFaceDatasetLoader class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a patcher for HfApi
        self.hf_api_patcher = mock.patch("ember.core.utils.data.base.loaders.HfApi")
        self.mock_hf_api_cls = self.hf_api_patcher.start()
        self.mock_hf_api = self.mock_hf_api_cls.return_value

        # Make sure dataset_info doesn't actually make API calls
        self.mock_hf_api.dataset_info = mock.MagicMock()

        # Create a patcher for load_dataset
        self.load_dataset_patcher = mock.patch(
            "ember.core.utils.data.base.loaders.load_dataset"
        )
        self.mock_load_dataset = self.load_dataset_patcher.start()

        # Create a patcher for os.makedirs
        self.makedirs_patcher = mock.patch("os.makedirs")
        self.mock_makedirs = self.makedirs_patcher.start()

        # Create a patcher for os.path.expanduser
        self.expanduser_patcher = mock.patch("os.path.expanduser")
        self.mock_expanduser = self.expanduser_patcher.start()
        self.mock_expanduser.return_value = "/mocked/home"

        # Create a patcher for enable_progress_bar
        self.enable_progress_bar_patcher = mock.patch(
            "ember.core.utils.data.base.loaders.enable_progress_bar"
        )
        self.mock_enable_progress_bar = self.enable_progress_bar_patcher.start()

        # Create a patcher for disable_progress_bar
        self.disable_progress_bar_patcher = mock.patch(
            "ember.core.utils.data.base.loaders.disable_progress_bar"
        )
        self.mock_disable_progress_bar = self.disable_progress_bar_patcher.start()

        # Create a patcher for enable_caching
        self.enable_caching_patcher = mock.patch(
            "ember.core.utils.data.base.loaders.enable_caching"
        )
        self.mock_enable_caching = self.enable_caching_patcher.start()

        # Create a patcher for disable_caching
        self.disable_caching_patcher = mock.patch(
            "ember.core.utils.data.base.loaders.disable_caching"
        )
        self.mock_disable_caching = self.disable_caching_patcher.start()

    def tearDown(self) -> None:
        """Tear down test fixtures."""
        self.hf_api_patcher.stop()
        self.load_dataset_patcher.stop()
        self.makedirs_patcher.stop()
        self.expanduser_patcher.stop()
        self.enable_progress_bar_patcher.stop()
        self.disable_progress_bar_patcher.stop()
        self.enable_caching_patcher.stop()
        self.disable_caching_patcher.stop()

    def test_init_default_cache_dir(self) -> None:
        """HuggingFaceDatasetLoader should create default cache directory when not provided."""
        # Arrange & Act
        loader = HuggingFaceDatasetLoader()

        # Assert
        expected_cache_dir = os.path.join(
            "/mocked/home", ".cache", "huggingface", "datasets"
        )
        self.assertEqual(expected_cache_dir, loader.cache_dir)
        self.mock_makedirs.assert_called_once_with(expected_cache_dir, exist_ok=True)

    def test_init_custom_cache_dir(self) -> None:
        """HuggingFaceDatasetLoader should use and create the provided cache directory."""
        # Arrange
        custom_cache_dir = "/custom/cache/dir"

        # Act
        loader = HuggingFaceDatasetLoader(cache_dir=custom_cache_dir)

        # Assert
        self.assertEqual(custom_cache_dir, loader.cache_dir)
        self.mock_makedirs.assert_called_once_with(custom_cache_dir, exist_ok=True)

    def test_load_success(self) -> None:
        """load() should successfully load a dataset with proper error handling."""
        # Arrange
        dataset_name = "test_dataset"
        config_name = "test_config"
        mock_dataset = mock.MagicMock(spec=DatasetDict)
        self.mock_load_dataset.return_value = mock_dataset

        loader = HuggingFaceDatasetLoader()

        # Act
        result = loader.load(dataset_name=dataset_name, config=config_name)

        # Assert
        self.assertEqual(mock_dataset, result)
        self.mock_hf_api.dataset_info.assert_called_once_with(dataset_name)
        self.mock_load_dataset.assert_called_once_with(
            path=dataset_name, name=config_name, cache_dir=loader.cache_dir
        )
        self.mock_enable_progress_bar.assert_called_once()
        self.mock_enable_caching.assert_called_once()
        self.mock_disable_caching.assert_called_once()
        self.mock_disable_progress_bar.assert_called_once()

    def test_load_dataset_not_found(self) -> None:
        """load() should raise ValueError when the dataset cannot be found."""
        # Arrange
        dataset_name = "nonexistent_dataset"
        self.mock_hf_api.dataset_info.side_effect = Exception("Dataset not found")

        loader = HuggingFaceDatasetLoader()

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            loader.load(dataset_name=dataset_name)

        # Verify error message
        self.assertIn(dataset_name, str(context.exception))
        self.mock_hf_api.dataset_info.assert_called_once_with(dataset_name)
        self.mock_load_dataset.assert_not_called()

    def test_load_http_error(self) -> None:
        """load() should raise RuntimeError when an HTTP error occurs during download."""
        # Arrange
        dataset_name = "error_dataset"
        http_error = HTTPError("http://example.com", 404, "Not Found", {}, None)

        # Configure the mock for the initial check to succeed
        self.mock_hf_api.dataset_info.return_value = mock.MagicMock()

        # Configure the mock to fail on load with HTTP error
        self.mock_load_dataset.side_effect = http_error

        loader = HuggingFaceDatasetLoader()

        # Act & Assert
        with self.assertRaises(RuntimeError) as context:
            loader.load(dataset_name=dataset_name)

        # Verify error message contains useful information
        self.assertIn(dataset_name, str(context.exception))
        self.assertIn("Failed to download", str(context.exception))

        # Verify correct cleanup
        self.mock_disable_caching.assert_called_once()
        self.mock_disable_progress_bar.assert_called_once()

    def test_load_unexpected_error(self) -> None:
        """load() should raise RuntimeError with informative message for unexpected errors."""
        # Arrange
        dataset_name = "error_dataset"
        unexpected_error = RuntimeError("Unexpected test error")

        # Configure the mock for the initial check to succeed
        self.mock_hf_api.dataset_info.return_value = mock.MagicMock()

        # Configure the mock to fail on load with unexpected error
        self.mock_load_dataset.side_effect = unexpected_error

        loader = HuggingFaceDatasetLoader()

        # Act & Assert
        with self.assertRaises(RuntimeError) as context:
            loader.load(dataset_name=dataset_name)

        # Verify error message contains useful information
        self.assertIn(dataset_name, str(context.exception))
        self.assertIn(str(unexpected_error), str(context.exception))

        # Verify correct cleanup
        self.mock_disable_caching.assert_called_once()
        self.mock_disable_progress_bar.assert_called_once()


if __name__ == "__main__":
    unittest.main()
