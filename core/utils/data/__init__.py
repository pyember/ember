from typing import List, Optional, Type, Union

from ember.core.utils.data.metadata_registry import DatasetMetadataRegistry
from ember.core.utils.data.loader_factory import DatasetLoaderFactory
from ember.core.utils.data.initialization import initialize_dataset_registry
from ember.core.utils.data.base.models import DatasetEntry, DatasetInfo, TaskType
from ember.core.utils.data.base.config import BaseDatasetConfig
from ember.core.utils.data.base.loaders import (
    HuggingFaceDatasetLoader,
    IDatasetLoader,
)
from ember.core.utils.data.base.validators import (
    DatasetValidator,
    IDatasetValidator,
)
from ember.core.utils.data.base.samplers import DatasetSampler, IDatasetSampler
from ember.core.utils.data.service import DatasetService
from ember.core.utils.data.base.preppers import IDatasetPrepper


def load_dataset_entries(
    *,
    dataset_name: str,
    config: Union[str, BaseDatasetConfig, None] = None,
    num_samples: Optional[int] = None,
) -> List[DatasetEntry]:
    """Load and prepare dataset entries using a high-level one-liner API.

    This function provides a fast-track method to load and process dataset entries
    with minimal boilerplate. The pipeline executes the following steps:
      1. Initializes the metadata registry and loader factory.
      2. Discovers and registers available dataset plugins.
      3. Retrieves the dataset metadata and its associated prepper class.
      4. Constructs default loader, validator, and sampler components.
      5. Executes the complete dataset preparation pipeline via the DatasetService.

    Example:
        from ember.core.utils.data import load_dataset_entries
        entries = load_dataset_entries(
            dataset_name="mmlu",
            config={"config_name": "abstract_algebra", "split": "dev"},
            num_samples=5
        )
        # 'entries' now contains the processed dataset entries.

    Args:
        dataset_name (str): Unique identifier for the dataset.
        config (Union[str, BaseDatasetConfig, None]):
            Optional configuration specifying dataset details or selection criteria.
            Can be either a configuration name (str) or an instance of BaseDatasetConfig.
        num_samples (Optional[int]): Optional maximum number of samples to retrieve.

    Returns:
        List[DatasetEntry]: A list of dataset entries after processing.

    Raises:
        ValueError: If the dataset metadata or its corresponding prepper cannot be found.
        TypeError: If arguments are not provided as named parameters.
    """
    # Initialize the metadata registry and loader factory.
    metadata_registry: DatasetMetadataRegistry = DatasetMetadataRegistry()
    loader_factory: DatasetLoaderFactory = DatasetLoaderFactory()
    initialize_dataset_registry(
        metadata_registry=metadata_registry, loader_factory=loader_factory
    )
    loader_factory.discover_and_register_plugins()

    # Retrieve dataset metadata and the corresponding prepper class.
    dataset_info: Optional[DatasetInfo] = metadata_registry.get(dataset_name)
    if dataset_info is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in metadata registry.")

    prepper_class: Optional[Type[IDatasetPrepper]] = loader_factory.get_prepper_class(
        dataset_name=dataset_name
    )
    if prepper_class is None:
        raise ValueError(
            f"Prepper for dataset '{dataset_name}' not found in loader factory."
        )

    # Instantiate the prepper using the provided configuration.
    prepper: IDatasetPrepper = prepper_class(config=config)

    # Construct default loader, validator, and sampler instances.
    loader: IDatasetLoader = HuggingFaceDatasetLoader()
    validator: IDatasetValidator = DatasetValidator()
    sampler: IDatasetSampler = DatasetSampler()

    # Build the dataset service and execute the data preparation pipeline.
    dataset_service: DatasetService = DatasetService(
        loader=loader, validator=validator, sampler=sampler
    )
    return dataset_service.load_and_prepare(
        dataset_info=dataset_info,
        prepper=prepper,
        config=config,
        num_samples=num_samples,
    )
