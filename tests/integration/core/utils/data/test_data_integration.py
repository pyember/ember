"""Integration tests for the data module.

These tests verify that the major components of the data module work together correctly.
"""

import os
import unittest
from typing import Any, Dict, List, Optional, Type, Union
from unittest import mock

from datasets import Dataset, DatasetDict

from src.ember.core.utils.data import load_dataset_entries, BaseDatasetConfig
from src.ember.core.utils.data.base.models import DatasetEntry, DatasetInfo, TaskType
from src.ember.core.utils.data.datasets_registry.mmlu import MMLUConfig
from src.ember.core.utils.data.datasets_registry.halueval import HaluEvalConfig


# Skip unless environment variable is set to run integration tests
should_run_integration = os.environ.get("RUN_INTEGRATION_TESTS", "").lower() in (
    "1",
    "true",
    "yes",
)
skip_integration_reason = (
    "Integration tests require RUN_INTEGRATION_TESTS=1 environment variable"
)


@unittest.skipIf(not should_run_integration, skip_integration_reason)
class TestDataIntegration(unittest.TestCase):
    """Integration tests for the data module.

    These tests require access to external resources like the Hugging Face Hub.
    They should be skipped during regular test runs and only executed manually
    or in environments with proper connectivity.

    To run these tests, set the RUN_INTEGRATION_TESTS environment variable:
    RUN_INTEGRATION_TESTS=1 python -m pytest tests/integration/core/utils/data/test_data_integration.py
    """

    def setUp(self) -> None:
        """Set up test fixtures and mocks for external services."""
        # Create a patch for the load_dataset function to avoid actual downloads
        self.load_dataset_patcher = mock.patch("datasets.load_dataset")
        self.mock_load_dataset = self.load_dataset_patcher.start()

        # Create mock dataset structures
        self.mock_mmlu_dataset = self._create_mock_mmlu_dataset()
        self.mock_halueval_dataset = self._create_mock_halueval_dataset()

        # Configure the mock load_dataset to return our mock datasets
        def load_dataset_side_effect(path, name=None, **kwargs):
            if path == "cais/mmlu":
                return self.mock_mmlu_dataset
            elif path == "pminervini/HaluEval":
                return self.mock_halueval_dataset
            else:
                # Raise an error for unknown datasets
                raise ValueError(f"Unknown dataset: {path}")

        self.mock_load_dataset.side_effect = load_dataset_side_effect

    def tearDown(self) -> None:
        """Tear down test fixtures."""
        self.load_dataset_patcher.stop()

    def _create_mock_mmlu_dataset(self) -> DatasetDict:
        """Create a mock MMLU dataset.

        Returns:
            DatasetDict: A mock dataset dictionary with 'dev' split.
        """
        # Create a mock Dataset for the dev split
        mock_dataset = mock.MagicMock(spec=Dataset)
        mock_dataset.__len__.return_value = 5

        # Create sample items
        mock_items = [
            {
                "question": f"Question {i}",
                "choices": [f"Choice {j}" for j in range(4)],
                "answer": i % 4,
                "subject": "abstract_algebra",
            }
            for i in range(5)
        ]

        # Configure the dataset to return items
        mock_dataset.__getitem__.side_effect = lambda idx: mock_items[idx]
        mock_dataset.__iter__.return_value = iter(mock_items)

        # Create a DatasetDict with the dev split
        mock_dataset_dict = mock.MagicMock(spec=DatasetDict)
        mock_dataset_dict.__getitem__.side_effect = lambda key: {
            "dev": mock_dataset
        }.get(key)
        mock_dataset_dict.__contains__.side_effect = lambda key: key == "dev"
        mock_dataset_dict.keys.return_value = ["dev"]

        return mock_dataset_dict

    def _create_mock_halueval_dataset(self) -> DatasetDict:
        """Create a mock HaluEval dataset.

        Returns:
            DatasetDict: A mock dataset dictionary with 'data' split.
        """
        # Create a mock Dataset for the data split
        mock_dataset = mock.MagicMock(spec=Dataset)
        mock_dataset.__len__.return_value = 5

        # Create sample items
        mock_items = [
            {
                "knowledge": f"Knowledge {i}",
                "question": f"Question {i}",
                "right_answer": f"Right answer {i}",
                "hallucinated_answer": f"Hallucinated answer {i}",
            }
            for i in range(5)
        ]

        # Configure the dataset to return items
        mock_dataset.__getitem__.side_effect = lambda idx: mock_items[idx]
        mock_dataset.__iter__.return_value = iter(mock_items)

        # Create a DatasetDict with the data split
        mock_dataset_dict = mock.MagicMock(spec=DatasetDict)
        mock_dataset_dict.__getitem__.side_effect = lambda key: {
            "data": mock_dataset
        }.get(key)
        mock_dataset_dict.__contains__.side_effect = lambda key: key == "data"
        mock_dataset_dict.keys.return_value = ["data"]

        return mock_dataset_dict

    def test_load_mmlu_dataset(self) -> None:
        """load_dataset_entries() should load and prepare MMLU datasets."""
        # Arrange
        config = MMLUConfig(config_name="abstract_algebra", split="dev")
        num_samples = 3

        # Act
        entries = load_dataset_entries(
            dataset_name="mmlu", config=config, num_samples=num_samples
        )

        # Assert
        self.assertEqual(num_samples, len(entries))

        # Verify entry structure
        for entry in entries:
            self.assertIsInstance(entry, DatasetEntry)
            self.assertTrue(entry.query)
            self.assertEqual(4, len(entry.choices))
            self.assertIn("correct_answer", entry.metadata)
            self.assertIn("subject", entry.metadata)
            self.assertIn("config_name", entry.metadata)

    def test_load_halueval_dataset(self) -> None:
        """load_dataset_entries() should load and prepare HaluEval datasets."""
        # Arrange
        config = HaluEvalConfig()  # Uses defaults: config_name="qa", split="data"
        num_samples = 2

        # Act
        entries = load_dataset_entries(
            dataset_name="halueval", config=config, num_samples=num_samples
        )

        # Assert
        # Each HaluEval item generates 2 entries (not hallucinated and hallucinated)
        self.assertEqual(num_samples * 2, len(entries))

        # Verify entry structure
        for entry in entries:
            self.assertIsInstance(entry, DatasetEntry)
            self.assertTrue(entry.query)
            self.assertEqual(2, len(entry.choices))
            self.assertIn("correct_answer", entry.metadata)

    def test_load_with_string_config(self) -> None:
        """load_dataset_entries() should handle string configuration names."""
        # Arrange & Act
        entries = load_dataset_entries(
            dataset_name="mmlu", config="abstract_algebra", num_samples=2
        )

        # Assert
        self.assertEqual(2, len(entries))
        for entry in entries:
            self.assertIsInstance(entry, DatasetEntry)


class TestRealDataIntegration(unittest.TestCase):
    """Integration tests with real data access.

    These tests connect to the actual Hugging Face Hub and download real datasets.
    They should be manually executed and not part of automated test runs.

    To run these tests, set both environment variables:
    RUN_INTEGRATION_TESTS=1 ALLOW_EXTERNAL_API_CALLS=1 python -m pytest tests/integration/core/utils/data/test_data_integration.py::TestRealDataIntegration
    """

    # Skip unless both environment variables are set
    should_run_real_api = should_run_integration and os.environ.get(
        "ALLOW_EXTERNAL_API_CALLS", ""
    ).lower() in ("1", "true", "yes")
    skip_real_api_reason = "Real API tests require both RUN_INTEGRATION_TESTS=1 and ALLOW_EXTERNAL_API_CALLS=1"

    @unittest.skipIf(not should_run_real_api, skip_real_api_reason)
    def test_real_mmlu_dataset(self) -> None:
        """load_dataset_entries() should load actual MMLU data from Hugging Face."""
        # Arrange
        config = MMLUConfig(config_name="abstract_algebra", split="dev")
        num_samples = 2

        # Act
        entries = load_dataset_entries(
            dataset_name="mmlu", config=config, num_samples=num_samples
        )

        # Assert
        self.assertEqual(num_samples, len(entries))

        # Print entry details for manual verification
        print("\nMMID entries:")
        for i, entry in enumerate(entries):
            print(f"Entry {i+1}:")
            print(f"  Query: {entry.query}")
            print(f"  Choices: {entry.choices}")
            print(f"  Metadata: {entry.metadata}")

    @unittest.skipIf(not should_run_real_api, skip_real_api_reason)
    def test_real_halueval_dataset(self) -> None:
        """load_dataset_entries() should load actual HaluEval data from Hugging Face."""
        # Arrange
        config = HaluEvalConfig()
        num_samples = 1  # Will generate 2 entries (not hallucinated and hallucinated)

        # Act
        entries = load_dataset_entries(
            dataset_name="halueval", config=config, num_samples=num_samples
        )

        # Assert
        self.assertEqual(num_samples * 2, len(entries))

        # Print entry details for manual verification
        print("\nHaluEval entries:")
        for i, entry in enumerate(entries):
            print(f"Entry {i+1}:")
            print(f"  Query: {entry.query}")
            print(f"  Choices: {entry.choices}")
            print(f"  Metadata: {entry.metadata}")


if __name__ == "__main__":
    unittest.main()
