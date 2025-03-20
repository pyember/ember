from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset

# Define a type alias for datasets: either a Hugging Face Dataset or a list of dictionaries.
DatasetType = Union[Dataset, List[Dict[str, Any]]]


class IDatasetSampler(ABC):
    """Interface for sampling datasets.

    This abstract base class defines a contract for implementations that perform
    sampling on datasets. Subclasses must override the sample() method to return
    a subset of the provided dataset.
    """

    @abstractmethod
    def sample(self, *, data: DatasetType, num_samples: Optional[int]) -> DatasetType:
        """Samples a subset of items from a dataset.

        If num_samples is None or not a positive integer, the original dataset is
        returned unmodified.

        Args:
            data (DatasetType): The dataset to sample from. This can be a Hugging Face
                Dataset or a list of dictionaries.
            num_samples (Optional[int]): The maximum number of samples to extract. If None
                or <= 0, no sampling is performed and the original dataset is returned.

        Returns:
            DatasetType: A dataset subset containing at most num_samples items.

        Raises:
            NotImplementedError: This method must be overridden by a subclass.
        """
        raise NotImplementedError()


class DatasetSampler(IDatasetSampler):
    """Concrete implementation of the dataset sampler interface.

    Provides a method to sample a subset of entries from datasets while preserving
    the original dataset structure.
    """

    def sample(self, *, data: DatasetType, num_samples: Optional[int]) -> DatasetType:
        """Samples the dataset based on the specified maximum number of items.

        If num_samples is None or not greater than 0, the original dataset is returned.
        For a Hugging Face Dataset, the .select() method is employed to preserve the
        dataset structure. For list-based datasets, slicing is used.

        Args:
            data (DatasetType): The dataset from which to extract samples. This can be a
                Hugging Face Dataset or a list of dictionaries.
            num_samples (Optional[int]): The number of samples to extract. Must be a positive
                integer; otherwise, the original dataset is returned.

        Returns:
            DatasetType: A subset of the dataset containing at most num_samples items.
        """
        if num_samples is None or num_samples <= 0:
            return data

        if isinstance(data, Dataset):
            sample_length: int = min(num_samples, len(data))
            return data.select(list(range(sample_length)))

        return data[:num_samples]
