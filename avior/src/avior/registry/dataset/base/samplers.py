from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional
from datasets import Dataset


class IDatasetSampler(ABC):
    @abstractmethod
    def sample(
        self, data: Union[Dataset, List[Dict[str, Any]]], num_samples: Optional[int]
    ) -> Union[Dataset, List[Dict[str, Any]]]:
        pass


class DatasetSampler(IDatasetSampler):
    def sample(
        self, data: Union[Dataset, List[Dict[str, Any]]], num_samples: Optional[int]
    ) -> Union[Dataset, List[Dict[str, Any]]]:
        if not num_samples:
            return data
        
        # If it’s a Hugging Face Dataset, use .select(...) to keep it as a
        # Dataset of row-wise dicts, instead of a dict of columns.
        if isinstance(data, Dataset):
            return data.select(range(min(num_samples, len(data))))

        # Otherwise, assume it’s a list of dicts or similar.
        return data[:num_samples]
