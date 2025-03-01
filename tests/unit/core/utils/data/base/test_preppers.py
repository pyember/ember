"""Unit tests for dataset preppers."""

import unittest
from typing import Dict, List, Any, Optional

from ember.core.utils.data.base.config import BaseDatasetConfig
from ember.core.utils.data.base.models import DatasetEntry
from ember.core.utils.data.base.preppers import IDatasetPrepper


class ConcreteDatasetPrepper(IDatasetPrepper):
    """Concrete implementation of IDatasetPrepper for testing purposes."""

    def get_required_keys(self) -> List[str]:
        """Return a list of required keys for test data items.

        Returns:
            List[str]: List containing 'id', 'question', and 'answer' strings.
        """
        return ["id", "question", "answer"]

    def create_dataset_entries(self, *, item: Dict[str, Any]) -> List[DatasetEntry]:
        """Create DatasetEntry instances from a test item.

        Args:
            item (Dict[str, Any]): The input item dictionary.

        Returns:
            List[DatasetEntry]: List containing a single DatasetEntry created from the item.
        """
        entry = DatasetEntry(
            query=item["question"],
            choices={},
            metadata={"id": item["id"], "answer": item["answer"]},
        )
        return [entry]


class TestIDatasetPrepper(unittest.TestCase):
    """Test cases for the IDatasetPrepper interface."""

    def test_interface_enforcement(self) -> None:
        """IDatasetPrepper should require implementation of abstract methods."""
        # Attempt to instantiate the abstract base class directly
        with self.assertRaises(TypeError):
            IDatasetPrepper()  # type: ignore

        # Create a subclass that doesn't implement abstract methods
        class IncompletePrepper(IDatasetPrepper):
            pass

        # Attempt to instantiate the incomplete subclass
        with self.assertRaises(TypeError):
            IncompletePrepper()  # type: ignore

        # Create another subclass that implements only get_required_keys
        class PartialPrepper(IDatasetPrepper):
            def get_required_keys(self) -> List[str]:
                return ["test"]

        # Attempt to instantiate the partial implementation
        with self.assertRaises(TypeError):
            PartialPrepper()  # type: ignore

        # Create a complete implementation
        class CompletePrepper(IDatasetPrepper):
            def get_required_keys(self) -> List[str]:
                return ["test"]

            def create_dataset_entries(
                self, *, item: Dict[str, Any]
            ) -> List[DatasetEntry]:
                return [DatasetEntry(query="test")]

        # Should instantiate without error
        prepper = CompletePrepper()
        self.assertIsInstance(prepper, IDatasetPrepper)

    def test_init_with_config(self) -> None:
        """IDatasetPrepper should store the provided configuration."""
        # Arrange
        config = BaseDatasetConfig(config_name="test", split="test_split")

        # Act
        prepper = ConcreteDatasetPrepper(config=config)

        # Assert
        self.assertEqual(config, prepper._config)

    def test_init_without_config(self) -> None:
        """IDatasetPrepper should set _config to None when none is provided."""
        # Arrange & Act
        prepper = ConcreteDatasetPrepper()

        # Assert
        self.assertIsNone(prepper._config)


class TestConcreteDatasetPrepper(unittest.TestCase):
    """Test cases for a concrete implementation of IDatasetPrepper."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.prepper = ConcreteDatasetPrepper()
        self.test_item = {
            "id": "test123",
            "question": "What is the capital of France?",
            "answer": "Paris",
        }

    def test_get_required_keys(self) -> None:
        """get_required_keys() should return the expected list of keys."""
        # Arrange & Act
        required_keys = self.prepper.get_required_keys()

        # Assert
        self.assertEqual(["id", "question", "answer"], required_keys)

    def test_create_dataset_entries(self) -> None:
        """create_dataset_entries() should convert an item to DatasetEntry instances."""
        # Arrange & Act
        entries = self.prepper.create_dataset_entries(item=self.test_item)

        # Assert
        self.assertEqual(1, len(entries))
        entry = entries[0]
        self.assertEqual(self.test_item["question"], entry.query)
        self.assertEqual({}, entry.choices)
        self.assertEqual(
            {"id": self.test_item["id"], "answer": self.test_item["answer"]},
            entry.metadata,
        )

    def test_create_dataset_entries_with_named_params(self) -> None:
        """create_dataset_entries() should require named parameters."""
        # This test verifies that the method enforces the use of named parameters
        # by checking that positional arguments are rejected

        # Arrange & Act & Assert
        with self.assertRaises(TypeError):
            # Attempt to call with positional argument - should fail
            self.prepper.create_dataset_entries(self.test_item)  # type: ignore


class TestPreppersWithConfig(unittest.TestCase):
    """Test cases for IDatasetPrepper implementations that use config objects."""

    def test_config_usage_in_subclass(self) -> None:
        """Subclasses should be able to access and use config attributes properly."""

        # Define a test prepper subclass that uses config
        class ConfigAwarePrepper(IDatasetPrepper):
            def __init__(self, config: Optional[BaseDatasetConfig] = None) -> None:
                super().__init__(config=config)
                # Extract config values for use in methods
                self.config_name = (
                    getattr(self._config, "config_name", None) if self._config else None
                )
                self.split = (
                    getattr(self._config, "split", None) if self._config else None
                )

            def get_required_keys(self) -> List[str]:
                return ["question"]

            def create_dataset_entries(
                self, *, item: Dict[str, Any]
            ) -> List[DatasetEntry]:
                # Use config values in the creation process
                entry = DatasetEntry(
                    query=item["question"],
                    metadata={"config_name": self.config_name, "split": self.split},
                )
                return [entry]

        # Arrange
        config = BaseDatasetConfig(config_name="test_config", split="test_split")
        prepper = ConfigAwarePrepper(config=config)
        test_item = {"question": "What is the capital of France?"}

        # Act
        entries = prepper.create_dataset_entries(item=test_item)

        # Assert
        self.assertEqual(1, len(entries))
        entry = entries[0]
        self.assertEqual(test_item["question"], entry.query)
        self.assertEqual("test_config", entry.metadata["config_name"])
        self.assertEqual("test_split", entry.metadata["split"])


if __name__ == "__main__":
    unittest.main()
