import logging
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any
from datasets import Dataset, DatasetDict

_LOGGER: logging.Logger = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)

# Type aliases for dataset input and output types.
DatasetInputType = Union[DatasetDict, Dataset, List[Dict[str, Any]]]
DatasetOutputType = Union[Dataset, List[Dict[str, Any]]]


class IDatasetValidator(ABC):
    """Interface for validating dataset structures and individual items.

    This abstract base class enforces a contract for implementations that
    validate the overall dataset structure, check for required keys in items,
    and validate individual dataset entries.
    """

    @abstractmethod
    def validate_structure(self, *, dataset: DatasetInputType) -> DatasetOutputType:
        """Validates the structure of the provided dataset.

        Args:
            dataset (DatasetInputType): The dataset to be validated. Can be a
                'Dataset', 'DatasetDict', or a list of dictionaries.

        Returns:
            DatasetOutputType: A validated dataset (either as a 'Dataset' or a list
                of dictionaries) that meets the required non-empty criteria.

        Raises:
            ValueError: If the dataset or its relevant split is empty.
            TypeError:  If the provided input is not one of the expected types.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_required_keys(
        self, *, item: Dict[str, Any], required_keys: List[str]
    ) -> None:
        """Validates that the dataset item includes all required keys.

        Args:
            item (Dict[str, Any]): The dataset item to validate.
            required_keys (List[str]): The list of keys that are required to be present.

        Raises:
            ValueError: If any required keys are missing.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_item(self, *, item: Dict[str, Any], required_keys: List[str]) -> None:
        """Validates that a dataset item is properly formed.

        Args:
            item (Dict[str, Any]): The dataset item to check.
            required_keys (List[str]): The list of keys required to be present and not None.

        Raises:
            TypeError: If the 'item' is not a dictionary.
            KeyError:  If any required key is either missing or has a None value.
        """
        raise NotImplementedError


class DatasetValidator(IDatasetValidator):
    """Concrete implementation for validating datasets and their entries.

    This class ensures that the dataset conforms to expected structural
    requirements and that individual items contain all necessary keys.
    """

    def validate_structure(self, *, dataset: DatasetInputType) -> DatasetOutputType:
        """Validates and returns a well-formed dataset based on its type.

        For a 'Dataset', it ensures that it is not empty. For a 'DatasetDict',
        it selects a split (preferring 'validation' if available) and ensures the
        selected split is non-empty. For a list of dictionaries, it ensures the
        list is not empty.

        Args:
            dataset (DatasetInputType): The dataset to validate.

        Returns:
            DatasetOutputType: A valid dataset in a supported format.

        Raises:
            ValueError: If the dataset or its chosen split is empty.
            TypeError:  If the input type is unsupported.
        """
        if isinstance(dataset, Dataset):
            if len(dataset) == 0:
                raise ValueError("The provided Dataset is empty.")
            return dataset
        elif isinstance(dataset, DatasetDict):
            if len(list(dataset.keys())) == 0:
                raise ValueError("The provided DatasetDict is empty.")
            split_name: str = (
                "validation"
                if "validation" in dataset
                else next(iter(dataset.keys()), None)
            )
            if not split_name:
                raise ValueError("The provided DatasetDict has no splits available.")
            split_data = dataset[split_name]
            if len(split_data) == 0:
                raise ValueError(f"The split '{split_name}' in DatasetDict is empty.")
            return split_data
        elif isinstance(dataset, list):
            if not dataset:
                raise ValueError("The provided list dataset is empty.")
            return dataset
        else:
            raise TypeError(
                f"Input dataset must be of type Dataset, DatasetDict, or list of dicts; got {type(dataset)}."
            )

    def validate_required_keys(
        self, *, item: Dict[str, Any], required_keys: List[str]
    ) -> None:
        """Checks that all required keys are present in the dataset item.

        Args:
            item (Dict[str, Any]): The dataset item to check.
            required_keys (List[str]): Keys that must exist in the item.

        Raises:
            ValueError: If any required keys are missing from the item.
        """
        missing_keys = set(required_keys) - set(item.keys())
        if missing_keys:
            raise ValueError(f"Dataset item is missing required keys: {missing_keys}")

    def validate_item(self, *, item: Dict[str, Any], required_keys: List[str]) -> None:
        """Ensures both the presence and non-nullity of required keys in a dataset item.

        Args:
            item (Dict[str, Any]): The dataset entry to validate.
            required_keys (List[str]): The keys that must be present and non-None.

        Raises:
            TypeError: If the 'item' is not of type dict.
            KeyError:  If any required key is missing or has a None value.
        """
        if not isinstance(item, dict):
            raise TypeError(
                f"Expected 'item' to be a dict but received type: {type(item)}."
            )
        missing_or_none_keys = [
            key for key in required_keys if key not in item or item[key] is None
        ]
        if missing_or_none_keys:
            raise KeyError(
                f"Missing or None value for required keys: {', '.join(missing_or_none_keys)}"
            )
