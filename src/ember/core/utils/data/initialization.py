import logging
from typing import Dict, List, Type

from ember.core.utils.data.base.models import DatasetInfo, TaskType
from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.datasets_registry.commonsense_qa import CommonsenseQAPrepper
from ember.core.utils.data.datasets_registry.halueval import HaluEvalPrepper
from ember.core.utils.data.datasets_registry.mmlu import MMLUPrepper
from ember.core.utils.data.datasets_registry.short_answer import ShortAnswerPrepper
from ember.core.utils.data.datasets_registry.truthful_qa import TruthfulQAPrepper
from ember.core.utils.data.loader_factory import DatasetLoaderFactory
from ember.core.utils.data.metadata_registry import DatasetMetadataRegistry

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def initialize_dataset_registry(
    *,
    metadata_registry: DatasetMetadataRegistry,
    loader_factory: DatasetLoaderFactory,
) -> None:
    """Initializes the dataset registry with known datasets.

    Registers predefined dataset metadata with the metadata registry and maps associated
    data preppers in the loader factory.

    Args:
        metadata_registry (DatasetMetadataRegistry): The registry for dataset metadata.
        loader_factory (DatasetLoaderFactory): The factory for registering dataset preppers.

    Returns:
        None
    """
    # Defining dataset metadata configurations.
    dataset_metadatas: List[DatasetInfo] = [
        DatasetInfo(
            name="truthful_qa",
            description="A dataset for measuring truthfulness.",
            source="truthful_qa",
            task_type=TaskType.MULTIPLE_CHOICE,
        ),
        DatasetInfo(
            name="mmlu",
            description="Massive Multitask Language Understanding dataset.",
            source="cais/mmlu",
            task_type=TaskType.MULTIPLE_CHOICE,
        ),
        DatasetInfo(
            name="commonsense_qa",
            description="A dataset for commonsense QA.",
            source="commonsense_qa",
            task_type=TaskType.MULTIPLE_CHOICE,
        ),
        DatasetInfo(
            name="halueval",
            description="Dataset for evaluating hallucination in QA.",
            source="pminervini/HaluEval",
            task_type=TaskType.BINARY_CLASSIFICATION,
        ),
    ]

    # Registering each dataset metadata using explicit keyword arguments.
    for dataset_info in dataset_metadatas:
        # The UnifiedDatasetRegistry uses register_new, not register
        # This compatibility fix checks which method exists and uses the appropriate one
        if hasattr(metadata_registry, "register"):
            metadata_registry.register(dataset_info=dataset_info)
        elif hasattr(metadata_registry, "register_new"):
            metadata_registry.register_new(name=dataset_info.name, info=dataset_info)

    # Defining mapping of dataset names to their corresponding prepper classes.
    prepper_mappings: Dict[str, Type[IDatasetPrepper]] = {
        "truthful_qa": TruthfulQAPrepper,
        "mmlu": MMLUPrepper,
        "commonsense_qa": CommonsenseQAPrepper,
        "halueval": HaluEvalPrepper,
        "my_shortanswer_ds": ShortAnswerPrepper,
    }

    # Registering each dataset prepper using explicit keyword arguments.
    for dataset_name, prepper_class in prepper_mappings.items():
        loader_factory.register(dataset_name=dataset_name, prepper_class=prepper_class)

    logger.info("Initialized dataset registry with known datasets.")
