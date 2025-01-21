import logging
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class IDatasetValidator(ABC):
    @abstractmethod
    def validate_structure(
        self, dataset: Union[DatasetDict, Dataset, List[Dict[str, Any]]]
    ) -> Union[Dataset, List[Dict[str, Any]]]:
        pass

    @abstractmethod
    def validate_required_keys(
        self, item: Dict[str, Any], required_keys: List[str]
    ) -> None:
        pass

    @abstractmethod
    def validate_item(self, item: Dict[str, Any], required_keys: List[str]) -> None:
        pass


class DatasetValidator(IDatasetValidator):
    def validate_structure(
        self, dataset: Union[DatasetDict, Dataset, List[Dict[str, Any]]]
    ) -> Union[Dataset, List[Dict[str, Any]]]:
        if isinstance(dataset, Dataset):
            if len(dataset) == 0:
                raise ValueError("Dataset is empty")
            return dataset
        elif isinstance(dataset, DatasetDict):
            if not dataset:
                raise ValueError("DatasetDict is empty.")
            split_name = (
                "validation"
                if "validation" in dataset
                else next(iter(dataset.keys()), None)
            )
            if not split_name:
                raise ValueError("DatasetDict has no splits.")
            split_data = dataset[split_name]
            if len(split_data) == 0:
                raise ValueError(f"Split '{split_name}' is empty.")
            return split_data
        elif isinstance(dataset, list):
            if not dataset:
                raise ValueError("Dataset is empty")
            return dataset
        else:
            raise TypeError("Input must be a Dataset, DatasetDict, or list of dicts")

    def validate_required_keys(
        self, item: Dict[str, Any], required_keys: List[str]
    ) -> None:
        missing_keys = set(required_keys) - set(item.keys())
        if missing_keys:
            raise ValueError(f"Dataset is missing required keys: {missing_keys}")

    def validate_item(self, item: Dict[str, Any], required_keys: List[str]) -> None:
        if not isinstance(item, dict):
            raise TypeError(f"Item must be a dictionary, got {type(item)}")
        missing_or_none = [k for k in required_keys if k not in item or item[k] is None]
        if missing_or_none:
            raise KeyError(
                f"Missing or None required keys: {', '.join(missing_or_none)}"
            )
