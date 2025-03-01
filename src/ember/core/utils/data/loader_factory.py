import logging
from typing import Dict, List, Optional, Type
from importlib.metadata import entry_points

from ember.core.utils.data.base.preppers import IDatasetPrepper

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def discover_preppers(
    *, entry_point_group: str = "ember.dataset_preppers"
) -> Dict[str, Type[IDatasetPrepper]]:
    """Discovers dataset prepper classes from the specified entry point group.

    Iterates over the entry points in the provided group, attempts to load each
    associated dataset prepper class, and returns a mapping of dataset names to their
    corresponding prepper classes.

    Args:
        entry_point_group (str): Entry point group from which to discover dataset
            preppers. Defaults to "ember.dataset_preppers".

    Returns:
        Dict[str, Type[IDatasetPrepper]]: Mapping of dataset names to corresponding
            dataset prepper classes.
    """
    discovered: Dict[str, Type[IDatasetPrepper]] = {}
    entry_points_obj = entry_points()
    for entry_point in entry_points_obj.select(group=entry_point_group):
        try:
            prepper_cls: Type[IDatasetPrepper] = entry_point.load()
            dataset_name: str = entry_point.name
            discovered[dataset_name] = prepper_cls
        except Exception as error:
            logger.warning(
                "Failed to load dataset prepper plugin for '%s': %s",
                entry_point.name,
                error,
                exc_info=True,
            )
    return discovered


class DatasetLoaderFactory:
    """Factory for managing dataset loader preppers.

    This class maintains a registry mapping dataset names to their associated dataset
    prepper classes. It provides methods to register, retrieve, list, clear, and automatically
    discover dataset preppers via entry points.
    """

    def __init__(self) -> None:
        """Initializes the DatasetLoaderFactory with an empty registry.

        The registry maps a dataset's unique name to its corresponding dataset prepper class.
        """
        self._registry: Dict[str, Type[IDatasetPrepper]] = {}

    def register(
        self, *, dataset_name: str, prepper_class: Type[IDatasetPrepper]
    ) -> None:
        """Registers a dataset prepper class for a given dataset.

        Args:
            dataset_name (str): Unique identifier for the dataset.
            prepper_class (Type[IDatasetPrepper]): The dataset prepper class to register.
        """
        self._registry[dataset_name] = prepper_class
        logger.info("Registered loader prepper for dataset: '%s'", dataset_name)

    def get_prepper_class(
        self, *, dataset_name: str
    ) -> Optional[Type[IDatasetPrepper]]:
        """Retrieves the registered dataset prepper class for the provided dataset name.

        Args:
            dataset_name (str): The unique name of the dataset.

        Returns:
            Optional[Type[IDatasetPrepper]]: The dataset prepper class if registered;
                None otherwise.
        """
        return self._registry.get(dataset_name)

    def list_registered_preppers(self) -> List[str]:
        """Lists all registered dataset names.

        Returns:
            List[str]: A list of dataset names that have registered preppers.
        """
        return list(self._registry.keys())

    def clear(self) -> None:
        """Clears all registered dataset preppers from the registry."""
        self._registry.clear()

    def discover_and_register_plugins(self) -> None:
        """Discovers and registers dataset prepper plugins automatically.

        Uses the entry point group "ember.dataset_preppers" to discover dataset prepper
        classes. Each discovered prepper is registered into the factory's registry using
        named method invocation.
        """
        discovered_preppers: Dict[str, Type[IDatasetPrepper]] = discover_preppers(
            entry_point_group="ember.dataset_preppers"
        )
        for dataset_name, prepper_cls in discovered_preppers.items():
            self.register(dataset_name=dataset_name, prepper_class=prepper_cls)
        logger.info(
            "Auto-registered plugin preppers: %s", list(discovered_preppers.keys())
        )
