import logging
from typing import List, Dict, Any, Optional

# 1) Import our dataset registry tools:
from src.avior.registry.dataset.registry.metadata_registry import DatasetMetadataRegistry
from src.avior.registry.dataset.registry.loader_factory import DatasetLoaderFactory
from src.avior.registry.dataset.registry.initialization import initialize_dataset_registry

# 2) Import or define dataset loader/validator/sampler:
# If you have existing ones, import them. For now, we'll assume defaults or mocks.
from src.avior.registry.dataset.base.loaders import HuggingFaceDatasetLoader, IDatasetLoader
from src.avior.registry.dataset.base.validators import IDatasetValidator
from src.avior.registry.dataset.base.samplers import IDatasetSampler
from src.avior.registry.dataset.base.models import DatasetInfo, DatasetEntry, TaskType
from src.avior.registry.dataset.base.preppers import IDatasetPrepper
from src.avior.registry.dataset.datasets.mmlu import MMLUConfig
from src.avior.registry.dataset.base.validators import DatasetValidator
from src.avior.registry.dataset.base.samplers import DatasetSampler
from src.avior.registry.dataset.datasets.halueval import HaluEvalConfig

# 3) Import the DatasetService to actually use the pipeline:
from src.avior.registry.dataset.registry.service import DatasetService


# ----------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 1) Create a metadata registry and loader factory:
    metadata_registry = DatasetMetadataRegistry()
    loader_factory = DatasetLoaderFactory()
    
    # 2) Initialize the registry with known “built-in” datasets:
    initialize_dataset_registry(metadata_registry, loader_factory)

    # 3) Optionally, discover any additional plugin-based preppers from pyproject.toml:
    loader_factory.discover_and_register_plugins()

    # 4) Retrieve dataset info from our registry (for example, "mmlu"):
    mmlu_info: Optional[DatasetInfo] = metadata_registry.get("mmlu")
    if not mmlu_info:
        raise ValueError("MMLU dataset not properly registered.")

    # 5) Obtain a prepper class from loader_factory:
    #    This is the class that knows how to format MMLU data into DatasetEntry objects.
    mmlu_prepper_class = loader_factory.get_prepper_class("mmlu")
    if not mmlu_prepper_class:
        raise ValueError("No MMLU prepper found. Make sure it's registered.")

    # 5a) Create an MMLUConfig specifying which sub-config and split you want:
    mmlu_config = MMLUConfig(config_name="abstract_algebra", split="dev")

    # 5b) Pass it into the MMLUPrepper constructor:
    mmlu_prepper: IDatasetPrepper = mmlu_prepper_class(config=mmlu_config)

    # 6) Construct a dataset loader, validator, and sampler:
    #    (Replace HuggingFaceDatasetLoader with your real loader if you have a custom approach.)
    loader: IDatasetLoader = HuggingFaceDatasetLoader()
    validator: IDatasetValidator = DatasetValidator()
    sampler: IDatasetSampler = DatasetSampler()

    # 7) Instantiate a DatasetService to handle load, validation, transform, sampling, and prep:
    dataset_service = DatasetService(
        loader=loader,
        validator=validator,
        sampler=sampler,
        transformers=[]  # Insert any specialized transformers if needed
    )

    # 8) Load and prepare the dataset:
    #    "mmlu" is a Hugging Face dataset name in the code snippet, but you’d use a real ID.
    logger.info(f"Loading and preparing dataset: {mmlu_info.name}")
    try:
        # Pass the full MMLUConfig object so both config_name and split are handled:
        dataset_entries: List[DatasetEntry] = dataset_service.load_and_prepare(
            dataset_info=mmlu_info,
            prepper=mmlu_prepper,
            config=mmlu_config,
            num_samples=5
        )
        
        # 9) Print or process these dataset entries:
        logger.info(f"Received {len(dataset_entries)} prepared entries for '{mmlu_info.name}'.")
        for i, entry in enumerate(dataset_entries):
            logger.info(f"Entry #{i+1}:\n{entry.model_dump_json(indent=2)}")
    
    except Exception as e:
        logger.error(f"Error during dataset preparation: {e}")

    # 9) Let's do the same for HaluEval:
    halu_info: Optional[DatasetInfo] = metadata_registry.get("halueval")
    if not halu_info:
        raise ValueError("HaluEval dataset not properly registered.")

    halu_prepper_class = loader_factory.get_prepper_class("halueval")
    if not halu_prepper_class:
        raise ValueError("No HaluEval prepper found. Make sure it's registered.")

    # Create config & prepper, defaulting to config_name="qa", split="data"
    halu_config = HaluEvalConfig()
    halu_prepper: IDatasetPrepper = halu_prepper_class(config=halu_config)

    logger.info(f"Loading and preparing dataset: {halu_info.name}")
    try:
        halu_dataset_entries: List[DatasetEntry] = dataset_service.load_and_prepare(
            dataset_info=halu_info,
            prepper=halu_prepper,
            config=halu_config,
            num_samples=3
        )
        logger.info(f"Received {len(halu_dataset_entries)} prepared entries for '{halu_info.name}'.")
        for i, entry in enumerate(halu_dataset_entries):
            logger.info(f"[HaluEval] Entry #{i+1}:\n{entry.model_dump_json(indent=2)}")
    except Exception as e:
        logger.error(f"Error during HaluEval dataset preparation: {e}")


if __name__ == "__main__":
    main()
