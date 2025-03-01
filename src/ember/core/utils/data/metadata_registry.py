from typing import Dict, Tuple, Optional, List, Type

from ember.core.utils.data.base.models import DatasetInfo
from ember.core.utils.data.base.preppers import IDatasetPrepper


class DatasetRegistry:
    """Registry mapping dataset names to their associated DatasetInfo and prepper instance.

    This class maintains a global mapping of datasets and their corresponding preppers.
    """

    _registry: Dict[str, Tuple[DatasetInfo, IDatasetPrepper]] = {}

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all entries from the dataset registry."""
        cls._registry.clear()

    @classmethod
    def register(cls, dataset_info: DatasetInfo, prepper: IDatasetPrepper) -> None:
        """Register a dataset with its corresponding prepper.

        Args:
            dataset_info (DatasetInfo): The metadata for the dataset.
            prepper (IDatasetPrepper): An instance of the dataset prepper.
        """
        cls._registry[dataset_info.name] = (dataset_info, prepper)

    @classmethod
    def get(cls, name: str) -> Optional[Tuple[DatasetInfo, IDatasetPrepper]]:
        """Retrieve the dataset metadata and its prepper by dataset name.

        Args:
            name (str): The name of the dataset.

        Returns:
            Optional[Tuple[DatasetInfo, IDatasetPrepper]]: A tuple containing the dataset metadata
            and the prepper instance if registered; otherwise, None.
        """
        return cls._registry.get(name)

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List the names of all registered datasets.

        Returns:
            List[str]: A list of dataset names.
        """
        return list(cls._registry.keys())


def register_dataset(
    name: str,
    description: str,
    source: str,
    task_type: str,
    prepper_class: Type[IDatasetPrepper],
) -> None:
    """Register a dataset with its metadata and associated prepper.

    This function constructs a DatasetInfo instance and registers it along with an
    instance of the provided prepper class in the global DatasetRegistry.

    Args:
        name (str): The name of the dataset.
        description (str): A brief description of the dataset.
        source (str): The source or provider of the dataset.
        task_type (str): The type of task associated with the dataset.
        prepper_class (Type[IDatasetPrepper]): The class of the dataset prepper. An instance of
            this class will be created and registered.
    """
    dataset_info: DatasetInfo = DatasetInfo(
        name=name,
        description=description,
        source=source,
        task_type=task_type,
    )
    DatasetRegistry.register(dataset_info=dataset_info, prepper=prepper_class())


class DatasetRegistryManager:
    """Manager for a custom dataset registry.

    This manager maintains its own mapping of datasets and their corresponding preppers,
    separate from the global DatasetRegistry.
    """

    def __init__(self) -> None:
        """Initialize the DatasetRegistryManager with an empty registry."""
        self._registry: Dict[str, Tuple[DatasetInfo, IDatasetPrepper]] = {}

    def register_dataset(
        self, info: DatasetInfo, prepper_class: Type[IDatasetPrepper]
    ) -> None:
        """Register a dataset and its associated prepper in the manager.

        Args:
            info (DatasetInfo): The dataset metadata.
            prepper_class (Type[IDatasetPrepper]): The prepper class to instantiate and register.
        """
        self._registry[info.name] = (info, prepper_class())

    def get_dataset_info(self, name: str) -> Optional[DatasetInfo]:
        """Retrieve the metadata for a specified dataset.

        Args:
            name (str): The name of the dataset.

        Returns:
            Optional[DatasetInfo]: The dataset metadata if it exists; otherwise, None.
        """
        result: Optional[Tuple[DatasetInfo, IDatasetPrepper]] = self._registry.get(name)
        return result[0] if result is not None else None

    def get_prepper_class(self, name: str) -> Optional[Type[IDatasetPrepper]]:
        """Retrieve the class of the prepper associated with the specified dataset.

        Args:
            name (str): The name of the dataset.

        Returns:
            Optional[Type[IDatasetPrepper]]: The dataset prepper class if registered; otherwise, None.
        """
        result: Optional[Tuple[DatasetInfo, IDatasetPrepper]] = self._registry.get(name)
        return type(result[1]) if result is not None else None

    def clear_all(self) -> None:
        """Clear all entries from the manager's dataset registry."""
        self._registry.clear()


class DatasetMetadataRegistry:
    """Registry for storing only dataset metadata (DatasetInfo).

    This lightweight registry is used primarily for initialization and does not handle
    dataset preppers.
    """

    def __init__(self) -> None:
        """Initialize an empty DatasetMetadataRegistry."""
        self._registry: Dict[str, DatasetInfo] = {}

    def register(self, dataset_info: DatasetInfo) -> None:
        """Register dataset metadata.

        Args:
            dataset_info (DatasetInfo): The dataset metadata to register.
        """
        self._registry[dataset_info.name] = dataset_info

    def get(self, name: str) -> Optional[DatasetInfo]:
        """Retrieve the metadata for a given dataset.

        Args:
            name (str): The name of the dataset.

        Returns:
            Optional[DatasetInfo]: The dataset metadata if found; otherwise, None.
        """
        return self._registry.get(name)

    def list_datasets(self) -> List[str]:
        """List the names of all datasets in the metadata registry.

        Returns:
            List[str]: A list of dataset names.
        """
        return list(self._registry.keys())
