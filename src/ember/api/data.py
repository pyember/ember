"""Data API for Ember.

This module is a facade for the Ember data API, providing simplified access to the
dataset utilities implemented in ember.core.utils.data.

Examples:
    # Loading a dataset
    from ember.api import datasets
    mmlu_data = datasets("mmlu")
    
    # Using the builder pattern
    from ember.api import DatasetBuilder
    dataset = DatasetBuilder().split("test").sample(100).build("mmlu")
    
    # Registering a custom dataset
    from ember.api import register, TaskType, DatasetEntry
    
    @register("custom_qa", source="custom/qa", task_type=TaskType.QUESTION_ANSWERING)
    class CustomQADataset:
        def load(self, config=None):
            return [
                DatasetEntry(
                    id="qa1",
                    content={
                        "question": "What is the capital of France?",
                        "answer": "Paris"
                    },
                    metadata={"category": "geography"}
                )
            ]
"""

# Import from core implementation
from ember.core.utils.data.base.models import TaskType, DatasetInfo, DatasetEntry
from ember.core.utils.data.base.config import BaseDatasetConfig as DatasetConfig
from ember.core.utils.data.registry import register, UNIFIED_REGISTRY
from typing import Dict, List, Optional, Union, Any, TypeVar, Generic, Type


# Type variables for generic typing
T = TypeVar("T")


class Dataset(Generic[T]):
    """A dataset representation with convenient access to entries.

    Attributes:
        entries: List of dataset entries
        info: Dataset metadata
    """

    def __init__(self, entries: List[T], info: Optional[DatasetInfo] = None):
        """Initialize a Dataset with entries and optional info.

        Args:
            entries: List of dataset entries
            info: Optional dataset metadata
        """
        self.entries = entries
        self.info = info

    def __getitem__(self, idx: int) -> T:
        """Get a dataset entry by index.

        Args:
            idx: Index of the entry to retrieve

        Returns:
            The dataset entry at the specified index

        Raises:
            IndexError: If the index is out of range
        """
        return self.entries[idx]

    def __iter__(self):
        """Iterate over dataset entries.

        Returns:
            Iterator over dataset entries
        """
        return iter(self.entries)

    def __len__(self) -> int:
        """Get the number of entries in the dataset.

        Returns:
            Number of entries
        """
        return len(self.entries)


class DatasetBuilder:
    """Builder for configuring dataset loading.

    Provides a fluent interface for setting dataset loading parameters.
    """

    def __init__(self):
        """Initialize a new DatasetBuilder with default configuration."""
        self._split = None
        self._sample_size = None
        self._seed = None
        self._config = {}

    def split(self, split_name: str) -> "DatasetBuilder":
        """Set the dataset split to load.

        Args:
            split_name: Name of the split (e.g., "train", "test", "validation")

        Returns:
            Self for method chaining
        """
        self._split = split_name
        return self

    def sample(self, count: int) -> "DatasetBuilder":
        """Set the number of samples to load.

        Args:
            count: Number of samples to load

        Returns:
            Self for method chaining
        """
        self._sample_size = count
        return self

    def seed(self, seed_value: int) -> "DatasetBuilder":
        """Set the random seed for reproducible sampling.

        Args:
            seed_value: Random seed value

        Returns:
            Self for method chaining
        """
        self._seed = seed_value
        return self

    def config(self, **kwargs) -> "DatasetBuilder":
        """Set additional configuration parameters.

        Args:
            **kwargs: Configuration parameters as keyword arguments

        Returns:
            Self for method chaining
        """
        self._config.update(kwargs)
        return self

    def build(self, dataset_name: str) -> Dataset[DatasetEntry]:
        """Build and load the dataset with the configured parameters.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            Loaded dataset

        Raises:
            ValueError: If the dataset is not found or loading fails
        """
        # Create a dataset config object
        config = DatasetConfig(
            split=self._split,
            sample_size=self._sample_size,
            random_seed=self._seed,
            **self._config,
        )

        # Load the dataset using the unified registry and service system
        return datasets(dataset_name, config=config)


def datasets(
    name: str, config: Optional[Union[Dict[str, Any], DatasetConfig]] = None
) -> Dataset[DatasetEntry]:
    """Load a dataset by name with optional configuration.

    This is the primary function for loading datasets in Ember.

    Args:
        name: Name of the dataset to load
        config: Optional configuration for dataset loading

    Returns:
        Loaded dataset

    Raises:
        ValueError: If the dataset is not found or loading fails
    """
    from ember.core.utils.data.service import DatasetService
    from ember.core.utils.data.base.loaders import HuggingFaceDatasetLoader
    from ember.core.utils.data.base.validators import BasicDatasetValidator
    from ember.core.utils.data.base.samplers import RandomDatasetSampler
    from ember.core.utils.data.base.transformers import NoOpTransformer

    # Get dataset registration info from the unified registry
    dataset_entry = UNIFIED_REGISTRY.get(name=name)
    if dataset_entry is None:
        raise ValueError(
            f"Dataset '{name}' not found. Available datasets: {UNIFIED_REGISTRY.list_datasets()}"
        )

    # Create the data service pipeline
    service = DatasetService(
        loader=HuggingFaceDatasetLoader(),
        validator=BasicDatasetValidator(),
        sampler=RandomDatasetSampler(),
        transformers=[NoOpTransformer()],
    )

    # Extract sample size from config if available
    sample_size = None
    if isinstance(config, DatasetConfig) and hasattr(config, "sample_size"):
        sample_size = config.sample_size

    # Load and prepare the dataset
    entries = service.load_and_prepare(
        dataset_info=dataset_entry.info,
        prepper=dataset_entry.prepper,
        config=config,
        num_samples=sample_size,
    )

    # Return the loaded dataset with its information
    return Dataset(entries=entries, info=dataset_entry.info)


def list_available_datasets() -> List[str]:
    """List all available datasets.

    Returns:
        Sorted list of available dataset names
    """
    return UNIFIED_REGISTRY.list_datasets()


def get_dataset_info(name: str) -> Optional[DatasetInfo]:
    """Get information about a dataset.

    Args:
        name: Name of the dataset

    Returns:
        Dataset information if found, None otherwise
    """
    return UNIFIED_REGISTRY.get_info(name=name)


__all__ = [
    # Primary API
    "datasets",
    "Dataset",
    "DatasetBuilder",
    "DatasetConfig",
    "list_available_datasets",
    "get_dataset_info",
    # Registration
    "register",
    # Core data types
    "DatasetInfo",
    "DatasetEntry",
    "TaskType",
]
