from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .models import DatasetEntry
from src.avior.registry.dataset.base.config import BaseDatasetConfig


class IDatasetPrepper(ABC):
    def __init__(self, config: Optional[BaseDatasetConfig] = None):
        """
        Datasets that require special parameters can pass a config object here.
        """
        self._config = config

    @abstractmethod
    def get_required_keys(self) -> List[str]:
        pass

    @abstractmethod
    def create_dataset_entries(self, item: Dict[str, Any]) -> List[DatasetEntry]:
        pass
