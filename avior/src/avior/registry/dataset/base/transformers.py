from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any
from datasets import Dataset


class IDatasetTransformer(ABC):
    @abstractmethod
    def transform(
        self, data: Union[Dataset, List[Dict[str, Any]]]
    ) -> Union[Dataset, List[Dict[str, Any]]]:
        pass


class NoOpTransformer(IDatasetTransformer):
    """A do-nothing transformer."""

    def transform(
        self, data: Union[Dataset, List[Dict[str, Any]]]
    ) -> Union[Dataset, List[Dict[str, Any]]]:
        return data
