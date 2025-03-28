"""Data API for Ember.

This module provides a facade for Ember's data processing system with a streamlined
interface for loading, transforming, and working with datasets.

Examples:
    # Loading a dataset directly
    from ember.api import datasets
    mmlu_data = datasets("mmlu")

    # Using the builder pattern with transformations
    from ember.api import DatasetBuilder
    dataset = (DatasetBuilder()
        .from_registry("mmlu")
        .subset("physics")
        .split("test")
        .sample(100)
        .transform(lambda x: {"query": f"Question: {x['question']}"})
        .build())

    # Accessing dataset entries
    for entry in dataset:
        print(f"Question: {entry.content['query']}")

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

from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from ember.core.utils.data.base.config import BaseDatasetConfig as DatasetConfig
from ember.core.utils.data.base.models import DatasetEntry, DatasetInfo, TaskType
from ember.core.utils.data.base.transformers import IDatasetTransformer
from ember.core.utils.data.registry import (
    DATASET_REGISTRY,
    initialize_registry,
    register,
)

# Initialize the registry when this module is imported
initialize_registry()

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
    """Builder for dataset loading configuration.

    Provides a fluent interface for specifying dataset parameters and transformations
    before loading. Enables method chaining for concise, readable dataset preparation.
    """

    def __init__(self) -> None:
        """Initialize dataset builder with default configuration."""
        self._dataset_name: Optional[str] = None
        self._split: Optional[str] = None
        self._sample_size: Optional[int] = None
        self._seed: Optional[int] = None
        self._config: Dict[str, Any] = {}
        self._transformers: List[IDatasetTransformer] = []

    def from_registry(self, dataset_name: str) -> "DatasetBuilder":
        """Specify dataset to load from registry.

        Args:
            dataset_name: Name of the registered dataset

        Returns:
            Self for method chaining

        Raises:
            ValueError: If dataset is not found in registry
        """
        if not DATASET_REGISTRY.get(name=dataset_name):
            available = DATASET_REGISTRY.list_datasets()
            raise ValueError(
                f"Dataset '{dataset_name}' not found. Available datasets: {available}"
            )
        self._dataset_name = dataset_name
        return self

    def subset(self, subset_name: str) -> "DatasetBuilder":
        """Select dataset subset.

        Args:
            subset_name: Name of the subset to select

        Returns:
            Self for method chaining
        """
        self._config["subset"] = subset_name
        return self

    def split(self, split_name: str) -> "DatasetBuilder":
        """Set dataset split.

        Args:
            split_name: Name of the split (e.g., "train", "test", "validation")

        Returns:
            Self for method chaining
        """
        self._split = split_name
        return self

    def sample(self, count: int) -> "DatasetBuilder":
        """Set number of samples to load.

        Args:
            count: Number of samples

        Returns:
            Self for method chaining

        Raises:
            ValueError: If count is negative
        """
        if count < 0:
            raise ValueError(f"Sample count must be non-negative, got {count}")
        self._sample_size = count
        return self

    def seed(self, seed_value: int) -> "DatasetBuilder":
        """Set random seed for reproducible sampling.

        Args:
            seed_value: Random seed value

        Returns:
            Self for method chaining
        """
        self._seed = seed_value
        return self

    def transform(
        self,
        transform_fn: Union[
            Callable[[Dict[str, Any]], Dict[str, Any]], IDatasetTransformer
        ],
    ) -> "DatasetBuilder":
        """Add transformation function to dataset processing pipeline.

        Transformations are applied in the order they're added.

        Args:
            transform_fn: Function that transforms dataset items or transformer instance

        Returns:
            Self for method chaining
        """
        # Import here to avoid circular imports
        from ember.core.utils.data.base.transformers import (
            DatasetType,
            IDatasetTransformer,
        )

        # Use transformer directly if it implements the interface
        if isinstance(transform_fn, IDatasetTransformer):
            self._transformers.append(transform_fn)
            return self

        # Create adapter for function-based transformers
        class FunctionTransformer(IDatasetTransformer):
            """Adapter converting functions to IDatasetTransformer."""

            def __init__(self, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
                """Initialize with transformation function.

                Args:
                    fn: Function that transforms dataset items
                """
                self._transform_fn = fn

            def transform(self, *, data: DatasetType) -> DatasetType:
                """Apply transformation to dataset.

                Args:
                    data: Dataset to transform

                Returns:
                    Transformed dataset
                """
                if hasattr(data, "map") and callable(getattr(data, "map", None)):
                    # For HuggingFace datasets
                    return data.map(self._transform_fn)  # type: ignore
                if isinstance(data, list):
                    # For list of dictionaries
                    return [self._transform_fn(item) for item in data]
                # For other data structures
                return self._transform_fn(data)  # type: ignore

        self._transformers.append(FunctionTransformer(transform_fn))
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

    def build(self, dataset_name: Optional[str] = None) -> Dataset[DatasetEntry]:
        """Build and load dataset with configured parameters.

        Args:
            dataset_name: Name of dataset to load (optional if set via from_registry)

        Returns:
            Loaded dataset with applied transformations

        Raises:
            ValueError: If dataset name is not provided or dataset not found
        """
        # Determine final dataset name
        final_name = dataset_name or self._dataset_name
        if not final_name:
            raise ValueError(
                "Dataset name must be provided either via build() or from_registry()"
            )

        # Handle Hugging Face dataset-specific configs
        # The key for MMLU and similar datasets is to pass a config name
        config_name = self._config.get("subset")

        # Create configuration object
        config = DatasetConfig(
            split=self._split,
            sample_size=self._sample_size,
            random_seed=self._seed,
            config_name=config_name,  # Pass as config_name to HF Dataset
            **self._config,
        )

        # Import service components
        from ember.core.utils.data.base.loaders import HuggingFaceDatasetLoader
        from ember.core.utils.data.base.samplers import DatasetSampler
        from ember.core.utils.data.base.validators import DatasetValidator
        from ember.core.utils.data.service import DatasetService

        # Get dataset entry from registry
        dataset_entry = DATASET_REGISTRY.get(name=final_name)
        if not dataset_entry:
            available = DATASET_REGISTRY.list_datasets()
            raise ValueError(
                f"Dataset '{final_name}' not found. Available datasets: {available}"
            )

        # Create data service with transformers
        service = DatasetService(
            loader=HuggingFaceDatasetLoader(),
            validator=DatasetValidator(),
            sampler=DatasetSampler(),
            transformers=self._transformers,
        )

        # Load and prepare dataset
        entries = service.load_and_prepare(
            dataset_info=dataset_entry.info,
            prepper=dataset_entry.prepper,
            config=config,
            num_samples=self._sample_size,
        )

        return Dataset(entries=entries, info=dataset_entry.info)


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
    from ember.core.utils.data.base.loaders import HuggingFaceDatasetLoader
    from ember.core.utils.data.base.samplers import DatasetSampler
    from ember.core.utils.data.base.transformers import NoOpTransformer
    from ember.core.utils.data.base.validators import DatasetValidator
    from ember.core.utils.data.service import DatasetService

    # Get dataset registration info from the registry
    dataset_entry = DATASET_REGISTRY.get(name=name)
    if dataset_entry is None:
        raise ValueError(
            f"Dataset '{name}' not found. Available: {DATASET_REGISTRY.list_datasets()}"
        )

    # Create the data service pipeline
    service = DatasetService(
        loader=HuggingFaceDatasetLoader(),
        validator=DatasetValidator(),
        sampler=DatasetSampler(),
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
    return DATASET_REGISTRY.list_datasets()


def get_dataset_info(name: str) -> Optional[DatasetInfo]:
    """Get information about a dataset.

    Args:
        name: Name of the dataset

    Returns:
        Dataset information if found, None otherwise
    """
    return DATASET_REGISTRY.get_info(name=name)


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
