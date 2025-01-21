from typing import Dict, Tuple, Optional, List
from src.avior.registry.dataset.base.models import DatasetInfo
from src.avior.registry.dataset.base.preppers import IDatasetPrepper

class DatasetRegistry:
    _registry: Dict[str, Tuple[DatasetInfo, IDatasetPrepper]] = {}

    @classmethod
    def clear_registry(cls) -> None:
        cls._registry = {}

    @classmethod
    def register(cls, dataset_info: DatasetInfo, prepper: IDatasetPrepper) -> None:
        cls._registry[dataset_info.name] = (dataset_info, prepper)

    @classmethod
    def get(cls, name: str) -> Optional[Tuple[DatasetInfo, IDatasetPrepper]]:
        return cls._registry.get(name)

    @classmethod
    def list_datasets(cls):
        return list(cls._registry.keys())

def register_dataset(
    name: str,
    description: str,
    source: str,
    task_type: str,
    prepper_class: type[IDatasetPrepper],
) -> None:
    dataset_info = DatasetInfo(
        name=name,
        description=description,
        source=source,
        task_type=task_type
    )
    DatasetRegistry.register(dataset_info, prepper_class())

class DatasetRegistryManager:
    def __init__(self) -> None:
        self._registry = {}

    def register_dataset(self, info: DatasetInfo, prepper_class: type[IDatasetPrepper]) -> None:
        self._registry[info.name] = (info, prepper_class())

    def get_dataset_info(self, name: str) -> Optional[DatasetInfo]:
        result = self._registry.get(name)
        return result[0] if result else None

    def get_prepper_class(self, name: str) -> Optional[type[IDatasetPrepper]]:
        result = self._registry.get(name)
        return type(result[1]) if result else None

    def clear_all(self) -> None:
        self._registry = {}

class DatasetMetadataRegistry:
    """
    A lightweight registry for just DatasetInfo, used by initialization.py.
    This does not handle Preppers at all. 
    """
    def __init__(self) -> None:
        self._registry: Dict[str, DatasetInfo] = {}

    def register(self, dataset_info: DatasetInfo) -> None:
        self._registry[dataset_info.name] = dataset_info

    def get(self, name: str) -> Optional[DatasetInfo]:
        return self._registry.get(name)

    def list_datasets(self) -> List[str]:
        return list(self._registry.keys())
