from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, TypedDict
from datasets import Dataset

# Define a type alias for dataset representations.
DatasetType = Union[Dataset, List[Dict[str, Any]]]

# Simple type alias for dataset items
DatasetItem = Dict[str, Any]


class IDatasetTransformer(ABC):
    """Interface for dataset transformers.

    This abstract base class defines the method specification for transforming dataset
    objects, ensuring consistency across transformer implementations.
    """

    @abstractmethod
    def transform(self, *, data: DatasetType) -> DatasetType:
        """Transforms the given dataset.

        Args:
            data (DatasetType): The input dataset, which can be either a HuggingFace Dataset
                or a list of dictionaries representing individual dataset entries.

        Returns:
            DatasetType: The transformed dataset.
        """
        raise NotImplementedError


class NoOpTransformer(IDatasetTransformer):
    """A no-operation transformer.

    This transformer performs no changes to the provided dataset and returns it as is.
    """

    def transform(self, *, data: DatasetType) -> DatasetType:
        """Returns the dataset without any modifications.

        Args:
            data (DatasetType): The input dataset to be returned unchanged.

        Returns:
            DatasetType: The same dataset provided as input.
        """
        return data
