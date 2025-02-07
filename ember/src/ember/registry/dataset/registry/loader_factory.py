import logging
from typing import Dict, List, Optional, Type
import pkg_resources

from src.ember.registry.dataset.base.preppers import IDatasetPrepper

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def discover_preppers(
    *, entry_point_group: str = "ember.dataset_preppers"
) -> Dict[str, Type[IDatasetPrepper]]:
    """Discovers dataset prepper classes from the specified entry point group.

    Iterates over the entry points in the given group, attempts to load each associated
    dataset prepper class, and returns a mapping of dataset names to their corresponding prepper classes.

    Args:
        entry_point_group: The entry point group from which to discover dataset preppers.
                           Defaults to "ember.dataset_preppers".

    Returns:
        A dictionary mapping dataset names (str) to dataset prepper classes (Type[IDatasetPrepper]).
    """
    discovered: Dict[str, Type[IDatasetPrepper]] = {}
    for entry_point in pkg_resources.iter_entry_points(group=entry_point_group):
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

    Maintains a registry mapping dataset names to their associated dataset prepper classes.
    Provides methods to register, retrieve, clear, and automatically discover dataset preppers
    via entry points.
    """

    def __init__(self) -> None:
        """Initializes the DatasetLoaderFactory with an empty registry."""
        self._registry: Dict[str, Type[IDatasetPrepper]] = {}

    def register(
        self, *, dataset_name: str, prepper_class: Type[IDatasetPrepper]
    ) -> None:
        """Registers a dataset prepper class for a given dataset.

        Args:
            dataset_name: The unique identifier for the dataset.
            prepper_class: The dataset prepper class to register.
        """
        self._registry[dataset_name] = prepper_class
        logger.info("Registered loader prepper for dataset: '%s'", dataset_name)

    def get_prepper_class(
        self, *, dataset_name: str
    ) -> Optional[Type[IDatasetPrepper]]:
        """Retrieves the registered dataset prepper class for the specified dataset.

        Args:
            dataset_name: The name of the dataset to look up.

        Returns:
            The dataset prepper class if found; otherwise, None.
        """
        return self._registry.get(dataset_name)

    def list_registered_preppers(self) -> List[str]:
        """Lists all registered dataset names.

        Returns:
            A list of dataset names (str) that have registered preppers.
        """
        return list(self._registry.keys())

    def clear(self) -> None:
        """Clears all registered dataset preppers from the registry."""
        self._registry.clear()

    def discover_and_register_plugins(self) -> None:
        """Discovers and automatically registers dataset prepper plugins.

        Uses the 'ember.dataset_preppers' entry point group to discover dataset preppers,
        then registers each discovered prepper into the registry using named method invocation.
        """
        discovered_preppers: Dict[str, Type[IDatasetPrepper]] = discover_preppers(
            entry_point_group="ember.dataset_preppers"
        )
        for dataset_name, prepper_cls in discovered_preppers.items():
            self.register(dataset_name=dataset_name, prepper_class=prepper_cls)
        logger.info(
            "Auto-registered plugin preppers: %s", list(discovered_preppers.keys())
        )
