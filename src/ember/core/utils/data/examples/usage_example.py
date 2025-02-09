import logging
from typing import List, Optional

from ember.core.utils.data.metadata_registry import DatasetMetadataRegistry
from ember.core.utils.data.loader_factory import DatasetLoaderFactory
from ember.core.utils.data.initialization import initialize_dataset_registry
from ember.core.utils.data.base.loaders import HuggingFaceDatasetLoader, IDatasetLoader
from ember.core.utils.data.base.validators import IDatasetValidator, DatasetValidator
from ember.core.utils.data.base.samplers import IDatasetSampler, DatasetSampler
from ember.core.utils.data.base.models import DatasetInfo, DatasetEntry, TaskType
from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.datasets_registry.mmlu import MMLUConfig
from ember.core.utils.data.datasets_registry.halueval import HaluEvalConfig
from ember.core.utils.data.service import DatasetService


def main() -> None:
    """Main entry point for loading, preparing, and logging dataset entries.

    This function initializes the dataset registry and loader factory, retrieves
    configurations for the MMLU and HaluEval datasets, instantiates their respective
    preppers, loads the datasets, and logs the prepared entries.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create the metadata registry and dataset loader factory.
    metadata_registry: DatasetMetadataRegistry = DatasetMetadataRegistry()
    loader_factory: DatasetLoaderFactory = DatasetLoaderFactory()

    # Initialize the registry with built-in datasets.
    initialize_dataset_registry(
        metadata_registry=metadata_registry, loader_factory=loader_factory
    )

    # Discover and register any additional plugin-based preppers.
    loader_factory.discover_and_register_plugins()

    # Retrieve MMLU dataset info from the registry.
    mmlu_info: Optional[DatasetInfo] = metadata_registry.get("mmlu")
    if mmlu_info is None:
        raise ValueError("MMLU dataset not properly registered.")

    # Obtain the prepper class for MMLU.
    mmlu_prepper_class = loader_factory.get_prepper_class("mmlu")
    if mmlu_prepper_class is None:
        raise ValueError("No MMLU prepper found. Make sure it's registered.")

    # Create the MMLU configuration and instantiate its prepper.
    mmlu_config: MMLUConfig = MMLUConfig(config_name="abstract_algebra", split="dev")
    mmlu_prepper: IDatasetPrepper = mmlu_prepper_class(config=mmlu_config)

    # Construct the dataset loader, validator, and sampler.
    loader: IDatasetLoader = HuggingFaceDatasetLoader()
    validator: IDatasetValidator = DatasetValidator()
    sampler: IDatasetSampler = DatasetSampler()

    # Instantiate the DatasetService with named method invocation.
    dataset_service: DatasetService = DatasetService(
        loader=loader,
        validator=validator,
        sampler=sampler,
        transformers=[],  # Insert any specialized transformers if needed.
    )

    # Load and prepare the MMLU dataset.
    logger.info("Loading and preparing dataset: %s", mmlu_info.name)
    try:
        dataset_entries: List[DatasetEntry] = dataset_service.load_and_prepare(
            dataset_info=mmlu_info,
            prepper=mmlu_prepper,
            config=mmlu_config,
            num_samples=5,
        )
        logger.info(
            "Received %d prepared entries for '%s'.",
            len(dataset_entries),
            mmlu_info.name,
        )
        for idx, entry in enumerate(dataset_entries, start=1):
            logger.info("MMLU Entry #%d:\n%s", idx, entry.model_dump_json(indent=2))
    except Exception as error:
        logger.exception("Error during MMLU dataset preparation: %s", error)

    # Retrieve HaluEval dataset info from the registry.
    halu_info: Optional[DatasetInfo] = metadata_registry.get("halueval")
    if halu_info is None:
        raise ValueError("HaluEval dataset not properly registered.")

    # Obtain the prepper class for HaluEval.
    halu_prepper_class = loader_factory.get_prepper_class("halueval")
    if halu_prepper_class is None:
        raise ValueError("No HaluEval prepper found. Make sure it's registered.")

    # Create the HaluEval configuration and instantiate its prepper.
    halu_config: HaluEvalConfig = (
        HaluEvalConfig()
    )  # Defaults: config_name="qa", split="data"
    halu_prepper: IDatasetPrepper = halu_prepper_class(config=halu_config)

    logger.info("Loading and preparing dataset: %s", halu_info.name)
    try:
        halu_dataset_entries: List[DatasetEntry] = dataset_service.load_and_prepare(
            dataset_info=halu_info,
            prepper=halu_prepper,
            config=halu_config,
            num_samples=3,
        )
        logger.info(
            "Received %d prepared entries for '%s'.",
            len(halu_dataset_entries),
            halu_info.name,
        )
        for idx, entry in enumerate(halu_dataset_entries, start=1):
            logger.info(
                "[HaluEval] Entry #%d:\n%s", idx, entry.model_dump_json(indent=2)
            )
    except Exception as error:
        logger.exception("Error during HaluEval dataset preparation: %s", error)


if __name__ == "__main__":
    main()
