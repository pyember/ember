from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

from .models import DatasetEntry
from ember.core.utils.data.base.config import BaseDatasetConfig
from ember.core.utils.data.base.transformers import DatasetItem


class IDatasetPrepper(ABC):
    """Interface for dataset preparation.

    This abstract base class defines the contract for preparing datasets by
    specifying required keys and generating dataset entries from individual
    items.

    Attributes:
        _config (Optional[BaseDatasetConfig]): Optional configuration object that
            provides specialized parameters needed for dataset preparation.
    """

    def __init__(self, config: Optional[BaseDatasetConfig] = None) -> None:
        """Initializes the dataset prepper with an optional configuration.

        Args:
            config (Optional[BaseDatasetConfig]): A configuration object providing
                dataset-specific parameters. Defaults to None.
        """
        self._config: Optional[BaseDatasetConfig] = config

    @abstractmethod
    def get_required_keys(self) -> List[str]:
        """Retrieves the list of keys required for each dataset item.

        Returns:
            List[str]: A list of strings representing the required keys.
        """
        raise NotImplementedError("Subclasses must implement get_required_keys()")

    @abstractmethod
    def create_dataset_entries(self, *, item: DatasetItem) -> List[DatasetEntry]:
        """Generates dataset entries from a given input item.

        This method expects its parameters to be passed as named arguments and
        must be implemented by subclasses to convert raw data into one or more
        DatasetEntry instances.

        Args:
            item (DatasetItem): A dictionary representing a single dataset item.

        Returns:
            List[DatasetEntry]: A list of dataset entries derived from the input item.
        """
        raise NotImplementedError("Subclasses must implement create_dataset_entries()")
