import logging
from typing import Dict, Type, Optional
from src.avior.registry.dataset.base.preppers import IDatasetPrepper
import pkg_resources  # or import importlib.metadata for py3.10+

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def discover_preppers(entry_point_group: str = "avior.dataset_preppers"):
    """
    Discovers dataset preppers registered via Python entry points
    and returns a dict of {dataset_name: prepper_class}.
    """
    discovered = {}
    for entry_point in pkg_resources.iter_entry_points(group=entry_point_group):
        try:
            cls = entry_point.load()
            dataset_name = entry_point.name
            discovered[dataset_name] = cls
        except Exception as e:
            logger.warning(
                f"Failed to load dataset prepper plugin for {entry_point.name}: {e}"
            )
    return discovered

class DatasetLoaderFactory:
    def __init__(self) -> None:
        self._registry: Dict[str, Type[IDatasetPrepper]] = {}

    def register(self, dataset_name: str, prepper_class: Type[IDatasetPrepper]) -> None:
        self._registry[dataset_name] = prepper_class
        logger.info(f"Registered loader prepper for dataset: {dataset_name}")

    def get_prepper_class(self, dataset_name: str) -> Optional[Type[IDatasetPrepper]]:
        return self._registry.get(dataset_name)

    def list_registered_preppers(self) -> list:
        return list(self._registry.keys())

    def clear(self) -> None:
        self._registry.clear()

    def discover_and_register_plugins(self):
        found_preppers = discover_preppers("avior.dataset_preppers")
        for ds_name, cls in found_preppers.items():
            self.register(ds_name, cls)
        logger.info(f"Auto-registered plugin preppers: {list(found_preppers.keys())}")
