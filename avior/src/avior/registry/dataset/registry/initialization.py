import logging
from src.avior.registry.dataset.registry.metadata_registry import (
    DatasetMetadataRegistry,
    register_dataset,
)
from src.avior.registry.dataset.registry.loader_factory import DatasetLoaderFactory
from src.avior.registry.dataset.base.models import TaskType, DatasetInfo

# Import your dataset-specific preppers:
from src.avior.registry.dataset.datasets.truthful_qa import TruthfulQAPrepper
from src.avior.registry.dataset.datasets.mmlu import MMLUPrepper
from src.avior.registry.dataset.datasets.commonsense_qa import CommonsenseQAPrepper
from src.avior.registry.dataset.datasets.halueval import HaluEvalPrepper
from src.avior.registry.dataset.datasets.short_answer import ShortAnswerPrepper
from src.avior.registry.dataset.datasets.code_prepper import CodePrepper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def initialize_dataset_registry(
    metadata_registry: DatasetMetadataRegistry, loader_factory: DatasetLoaderFactory
) -> None:
    # Example registrations:
    truthful_info = DatasetInfo(
        name="truthful_qa",
        description="A dataset for measuring truthfulness.",
        source="truthful_qa",
        task_type=TaskType.MULTIPLE_CHOICE,
    )
    mmlu_info = DatasetInfo(
        name="mmlu",
        description="Massive Multitask Language Understanding dataset.",
        source="cais/mmlu",
        task_type=TaskType.MULTIPLE_CHOICE,
    )
    commonsense_info = DatasetInfo(
        name="commonsense_qa",
        description="A dataset for commonsense QA.",
        source="commonsense_qa",
        task_type=TaskType.MULTIPLE_CHOICE,
    )
    halueval_info = DatasetInfo(
        name="halueval",
        description="Dataset for evaluating hallucination in QA.",
        source="pminervini/HaluEval",
        task_type=TaskType.BINARY_CLASSIFICATION,
    )

    # Register metadata
    metadata_registry.register(truthful_info)
    metadata_registry.register(mmlu_info)
    metadata_registry.register(commonsense_info)
    metadata_registry.register(halueval_info)

    # Register preppers
    loader_factory.register("truthful_qa", TruthfulQAPrepper)
    loader_factory.register("mmlu", MMLUPrepper)
    loader_factory.register("commonsense_qa", CommonsenseQAPrepper)
    loader_factory.register("halueval", HaluEvalPrepper)
    loader_factory.register("my_shortanswer_ds", ShortAnswerPrepper)
    loader_factory.register("my_code_ds", CodePrepper)

    logger.info("Initialized dataset registry with known datasets.")
