"""Dataset Registry Module

Registry for datasets with their metadata and preppers.
"""

import importlib
import logging
import pkgutil
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

from ember.core.utils.data.base.models import DatasetInfo, TaskType
from ember.core.utils.data.base.preppers import IDatasetPrepper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredDataset:
    """A registered dataset with its metadata and prepper."""

    name: str
    dataset_cls: Optional[Type[Any]] = None
    info: Optional[DatasetInfo] = None
    prepper: Optional[IDatasetPrepper] = None


class DatasetRegistry:
    """Registry for datasets with their metadata and preppers."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._registry: Dict[str, RegisteredDataset] = {}

    def register(
        self,
        *,
        name: str,
        dataset_cls: Optional[Type[Any]] = None,
        info: Optional[DatasetInfo] = None,
        prepper: Optional[IDatasetPrepper] = None,
    ) -> None:
        """Register a dataset.

        Args:
            name: Name of the dataset to register.
            dataset_cls: Optional class implementing the dataset.
            info: Optional dataset metadata information.
            prepper: Optional dataset prepper instance.
        """
        if name in self._registry:
            logger.warning(
                "Dataset %s is already registered; overwriting.",
                name,
            )
        self._registry[name] = RegisteredDataset(
            name=name,
            dataset_cls=dataset_cls,
            info=info,
            prepper=prepper,
        )
        logger.debug("Registered dataset: %s", name)

    def register_metadata(
        self,
        *,
        name: str,
        description: str,
        source: str,
        task_type: TaskType,
        prepper_class: Type[IDatasetPrepper],
    ) -> None:
        """Register dataset metadata with an associated prepper.

        Args:
            name: Name of the dataset.
            description: Brief description of the dataset.
            source: Source of the dataset.
            task_type: Type of task the dataset is for.
            prepper_class: Class to create a prepper instance from.
        """
        info = DatasetInfo(
            name=name, description=description, source=source, task_type=task_type
        )
        prepper_instance = prepper_class()
        self.register(name=name, info=info, prepper=prepper_instance)

    def get(self, *, name: str) -> Optional[RegisteredDataset]:
        """Retrieve a registered dataset by name.

        Args:
            name: Name of the dataset to retrieve.

        Returns:
            The registered dataset entry if found, or None.
        """
        return self._registry.get(name)

    def list_datasets(self) -> List[str]:
        """List all registered dataset names.

        Returns:
            Sorted list of all registered dataset names.
        """
        return sorted(self._registry.keys())

    def find(self, *, name: str) -> Optional[RegisteredDataset]:
        """Find a dataset by name.

        Args:
            name: Name of the dataset to find.

        Returns:
            The registered dataset entry if found, or None.
        """
        return self.get(name=name)

    def discover_datasets(self, *, package_name: str = "ember.data.datasets") -> None:
        """Discover and register datasets in the specified package.

        Args:
            package_name: Package to search for datasets.
        """
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            logger.warning("Could not import package: %s", package_name)
            return

        for _, mod_name, is_pkg in pkgutil.iter_modules(
            package.__path__, package.__name__ + "."
        ):
            try:
                importlib.import_module(mod_name)
                logger.debug("Imported module: %s", mod_name)
                if is_pkg:
                    self.discover_datasets(package_name=mod_name)
            except ImportError as error:
                logger.warning("Failed to import module %s: %s", mod_name, error)

    def get_info(self, *, name: str) -> Optional[DatasetInfo]:
        """Get metadata information for a registered dataset.

        Args:
            name: Name of the dataset.

        Returns:
            Dataset information if found, or None.
        """
        dataset = self.get(name=name)
        return dataset.info if dataset is not None else None

    def register_with_decorator(
        self,
        *,
        name: str,
        source: str,
        task_type: TaskType,
        description: str = "Custom dataset",
    ) -> Callable[[Type[Any]], Type[Any]]:
        """Decorator for registering a dataset class.

        Args:
            name: Name of the dataset.
            source: Source of the dataset.
            task_type: Type of task the dataset is for.
            description: Description of the dataset.

        Returns:
            Decorator function that registers the decorated class.
        """

        def decorator(cls: Type[Any]) -> Type[Any]:
            """Register a dataset class when decorated.

            Args:
                cls: Class to register.

            Returns:
                The decorated class.
            """
            if not hasattr(cls, "info"):
                cls.info = DatasetInfo(
                    name=name,
                    source=source,
                    task_type=task_type,
                    description=description,
                )
            self.register(name=name, dataset_cls=cls, info=cls.info)
            return cls

        return decorator

    def clear(self) -> None:
        """Clear all registered datasets."""
        self._registry.clear()
        logger.debug("Cleared all registered datasets.")


# Global singleton for dataset registry
DATASET_REGISTRY = DatasetRegistry()


# Decorator for registering datasets
def register(
    name: str, *, source: str, task_type: TaskType, description: str = "Custom dataset"
) -> Callable[[Type[Any]], Type[Any]]:
    """Decorator for registering a dataset class with the registry.

    Args:
        name: Name of the dataset.
        source: Source of the dataset.
        task_type: Type of task the dataset is for.
        description: Optional description of the dataset.

    Returns:
        Decorator function that registers the decorated class.
    """
    return DATASET_REGISTRY.register_with_decorator(
        name=name, source=source, task_type=task_type, description=description
    )


# Initialize the registry with core datasets
def initialize_registry() -> None:
    """Initialize the dataset registry with core datasets."""
    # Register core datasets
    from ember.core.utils.data.datasets_registry import (
        aime,
        codeforces,
        commonsense_qa,
        gpqa,
        halueval,
        mmlu,
        short_answer,
        truthful_qa,
    )

    # Register preppers from the core registry
    DATASET_REGISTRY.register_metadata(
        name="truthful_qa",
        description="TruthfulQA dataset",
        source="truthful_qa",
        task_type=TaskType.MULTIPLE_CHOICE,
        prepper_class=truthful_qa.TruthfulQAPrepper,
    )

    DATASET_REGISTRY.register_metadata(
        name="mmlu",
        description="Massive Multitask Language Understanding dataset",
        source="cais/mmlu",
        task_type=TaskType.MULTIPLE_CHOICE,
        prepper_class=mmlu.MMLUPrepper,
    )

    DATASET_REGISTRY.register_metadata(
        name="commonsense_qa",
        description="CommonsenseQA dataset",
        source="commonsense_qa",
        task_type=TaskType.MULTIPLE_CHOICE,
        prepper_class=commonsense_qa.CommonsenseQAPrepper,
    )

    DATASET_REGISTRY.register_metadata(
        name="halueval",
        description="HaluEval dataset",
        source="pminervini/HaluEval",
        task_type=TaskType.MULTIPLE_CHOICE,
        prepper_class=halueval.HaluEvalPrepper,
    )

    DATASET_REGISTRY.register_metadata(
        name="my_shortanswer_ds",
        description="Short Answer dataset",
        source="short_answer",
        task_type=TaskType.SHORT_ANSWER,
        prepper_class=short_answer.ShortAnswerPrepper,
    )

    # Register new datasets
    DATASET_REGISTRY.register_metadata(
        name="aime",
        description="American Invitational Mathematics Examination",
        source="Maxwell-Jia/AIME_2024",
        task_type=TaskType.SHORT_ANSWER,
        prepper_class=aime.AIMEPrepper,
    )

    DATASET_REGISTRY.register_metadata(
        name="gpqa",
        description="Graduate-level PhD science questions (Diamond subset)",
        source="Idavidrein/gpqa",
        task_type=TaskType.MULTIPLE_CHOICE,
        prepper_class=gpqa.GPQAPrepper,
    )

    DATASET_REGISTRY.register_metadata(
        name="codeforces",
        description="Competitive programming problems",
        source="open-r1/codeforces",
        task_type=TaskType.CODE_COMPLETION,
        prepper_class=codeforces.CodeForcesPrepper,
    )

    # Discover datasets in the ember.data.datasets package
    DATASET_REGISTRY.discover_datasets()
