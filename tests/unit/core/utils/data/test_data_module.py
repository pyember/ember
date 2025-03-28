"""Unit tests for the data module's high-level API."""

import unittest

# Import the function under test directly
from ember.core.utils.data import load_dataset_entries

# Import the types we need directly


class TestLoadDatasetEntriesBasic(unittest.TestCase):
    """Basic tests for the load_dataset_entries function that don't need complex mocking."""

    def test_load_dataset_entries_with_named_params(self) -> None:
        """load_dataset_entries() should require named parameters."""
        # Arrange & Act & Assert
        # This test doesn't need mocking as it fails before any functionality is executed
        with self.assertRaises(TypeError):
            load_dataset_entries("test_dataset")  # type: ignore


if __name__ == "__main__":
    unittest.main()
