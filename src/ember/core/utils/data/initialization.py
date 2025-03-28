"""Dataset initialization module.

This module provides functions for initializing the dataset registry with known datasets.
It maintains backward compatibility with the legacy registry system by implementing
a shim that delegates to the unified registry.
"""

import logging
from typing import Dict, List, Optional, Type

from ember.core.utils.data.base.models import DatasetInfo, TaskType
from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.loader_factory import DatasetLoaderFactory
from ember.core.utils.data.registry import (
    DATASET_REGISTRY,
    DatasetRegistry,
    initialize_registry,
)

logger: logging.Logger = logging.getLogger(__name__)


def initialize_dataset_registry(
    *,
    metadata_registry: DatasetRegistry = DATASET_REGISTRY,
    loader_factory: DatasetLoaderFactory,
) -> None:
    """Initializes the dataset registry with known datasets.

    Compatibility layer for the legacy initialization pattern that redirects
    to the unified registry implementation. This ensures consistent dataset registration
    regardless of which initialization function is called.

    Args:
        metadata_registry: The registry for dataset metadata.
            Defaults to the global DATASET_REGISTRY.
        loader_factory: The factory for registering dataset preppers.

    Returns:
        None
    """
    # Import all dataset-related modules and their prepper classes
    from ember.core.utils.data.datasets_registry.aime import AIMEPrepper
    from ember.core.utils.data.datasets_registry.codeforces import CodeForcesPrepper
    from ember.core.utils.data.datasets_registry.commonsense_qa import (
        CommonsenseQAPrepper,
    )
    from ember.core.utils.data.datasets_registry.gpqa import GPQAPrepper
    from ember.core.utils.data.datasets_registry.halueval import HaluEvalPrepper
    from ember.core.utils.data.datasets_registry.mmlu import MMLUPrepper
    from ember.core.utils.data.datasets_registry.short_answer import ShortAnswerPrepper
    from ember.core.utils.data.datasets_registry.truthful_qa import TruthfulQAPrepper

    # Define datasets with their metadata and corresponding prepper class
    # This directly recreates the same dataset registrations from the original
    datasets = [
        {
            "name": "truthful_qa",
            "description": "A dataset for measuring truthfulness.",
            "source": "truthful_qa",
            "task_type": TaskType.MULTIPLE_CHOICE,
            "prepper_class": TruthfulQAPrepper,
        },
        {
            "name": "mmlu",
            "description": "Massive Multitask Language Understanding dataset.",
            "source": "cais/mmlu",
            "task_type": TaskType.MULTIPLE_CHOICE,
            "prepper_class": MMLUPrepper,
        },
        {
            "name": "commonsense_qa",
            "description": "A dataset for commonsense QA.",
            "source": "commonsense_qa",
            "task_type": TaskType.MULTIPLE_CHOICE,
            "prepper_class": CommonsenseQAPrepper,
        },
        {
            "name": "halueval",
            "description": "Dataset for evaluating hallucination in QA.",
            "source": "pminervini/HaluEval",
            "task_type": TaskType.BINARY_CLASSIFICATION,
            "prepper_class": HaluEvalPrepper,
        },
        {
            "name": "aime",
            "description": "American Invitational Mathematics Examination",
            "source": "Maxwell-Jia/AIME_2024",
            "task_type": TaskType.SHORT_ANSWER,
            "prepper_class": AIMEPrepper,
        },
        {
            "name": "gpqa",
            "description": "Graduate-level PhD science questions (Diamond subset)",
            "source": "Idavidrein/gpqa",
            "task_type": TaskType.MULTIPLE_CHOICE,
            "prepper_class": GPQAPrepper,
        },
        {
            "name": "codeforces",
            "description": "Competitive programming problems",
            "source": "open-r1/codeforces",
            "task_type": TaskType.CODE_COMPLETION,
            "prepper_class": CodeForcesPrepper,
        },
        # Special case for ShortAnswerPrepper with different registration name
        {
            "name": "my_shortanswer_ds",
            "description": "Short Answer dataset",
            "source": "short_answer",
            "task_type": TaskType.SHORT_ANSWER,
            "prepper_class": ShortAnswerPrepper,
        },
    ]

    # Register each dataset with both the metadata registry and loader factory
    for dataset_info in datasets:
        name = dataset_info["name"]
        # Register metadata
        metadata_registry.register(
            name=name,
            info=DatasetInfo(
                name=name,
                description=dataset_info["description"],
                source=dataset_info["source"],
                task_type=dataset_info["task_type"],
            ),
        )
        # Register prepper
        loader_factory.register(
            dataset_name=name, prepper_class=dataset_info["prepper_class"]
        )

    logger.info("Initialized dataset registry with %d datasets", len(datasets))
