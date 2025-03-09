"""Unified Dataset Registry Module

This module provides a unified registry for datasets, consolidating legacy and new
registrations along with associated metadata and preppers.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Type

from ember.core.utils.data.base.models import DatasetInfo as LegacyDatasetInfo
from ember.core.utils.data.base.models import TaskType
from ember.core.utils.data.base.preppers import IDatasetPrepper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredDataset:
    """Dataclass representing a registered dataset."""
    name: str
    dataset_cls: Optional[Type[Any]] = None
    info: Optional[LegacyDatasetInfo] = None
    prepper: Optional[IDatasetPrepper] = None
    is_legacy: bool = False


class UnifiedDatasetRegistry:
    """A unified registry for datasets, supporting both new and legacy registrations."""

    def __init__(self) -> None:
        self._registry: Dict[str, RegisteredDataset] = {}
        self._legacy_registry: Dict[str, RegisteredDataset] = {}

    def register_new(
        self,
        *,
        name: str,
        dataset_cls: Optional[Type[Any]] = None,
        info: Optional[LegacyDatasetInfo] = None,
        prepper: Optional[IDatasetPrepper] = None
    ) -> None:
        """Register a new dataset.
        
        Args:
            name: Name of the dataset to register.
            dataset_cls: Optional class implementing the dataset.
            info: Optional dataset metadata information.
            prepper: Optional dataset prepper instance.
        """
        if name in self._registry:
            logger.warning("Dataset %s is already registered in the new registry; overwriting.", name)
        self._registry[name] = RegisteredDataset(
            name=name,
            dataset_cls=dataset_cls,
            info=info,
            prepper=prepper,
            is_legacy=False
        )
        logger.debug("Registered new dataset: %s", name)

    def register_legacy(
        self,
        *,
        name: str,
        dataset_cls: Optional[Type[Any]] = None,
        info: Optional[LegacyDatasetInfo] = None,
        prepper: Optional[IDatasetPrepper] = None
    ) -> None:
        """Register a legacy dataset.
        
        Args:
            name: Name of the legacy dataset to register.
            dataset_cls: Optional class implementing the dataset.
            info: Optional dataset metadata information.
            prepper: Optional dataset prepper instance.
        """
        if name in self._legacy_registry:
            logger.warning("Legacy dataset %s is already registered; overwriting.", name)
        self._legacy_registry[name] = RegisteredDataset(
            name=name,
            dataset_cls=dataset_cls,
            info=info,
            prepper=prepper,
            is_legacy=True
        )
        logger.debug("Registered legacy dataset: %s", name)

    def register_metadata(
        self,
        *,
        name: str,
        description: str,
        source: str,
        task_type: TaskType,
        prepper_class: Type[IDatasetPrepper]
    ) -> None:
        """Register dataset metadata with an associated prepper.
        
        Args:
            name: Name of the dataset.
            description: Brief description of the dataset.
            source: Source of the dataset.
            task_type: Type of task the dataset is for.
            prepper_class: Class to create a prepper instance from.
        """
        info: LegacyDatasetInfo = LegacyDatasetInfo(
            name=name,
            description=description,
            source=source,
            task_type=task_type
        )
        prepper_instance: IDatasetPrepper = prepper_class()
        self.register_new(name=name, info=info, prepper=prepper_instance)

    def get(self, *, name: str) -> Optional[RegisteredDataset]:
        """Retrieve a registered dataset by name.
        
        Args:
            name: Name of the dataset to retrieve.
            
        Returns:
            The registered dataset entry if found, or None.
        """
        dataset: Optional[RegisteredDataset] = self._registry.get(name)
        if dataset is None:
            dataset = self._legacy_registry.get(name)
        return dataset

    def list_datasets(self) -> List[str]:
        """List all registered dataset names.
        
        Returns:
            Sorted list of all registered dataset names.
        """
        all_names = set(self._registry.keys()).union(set(self._legacy_registry.keys()))
        return sorted(all_names)

    @lru_cache(maxsize=128)
    def find(self, *, name: str) -> Optional[RegisteredDataset]:
        """Find a dataset by name, checking both new and legacy registries.
        
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

        for _, mod_name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            try:
                importlib.import_module(mod_name)
                logger.debug("Imported module: %s", mod_name)
                if is_pkg:
                    self.discover_datasets(package_name=mod_name)
            except ImportError as error:
                logger.warning("Failed to import module %s: %s", mod_name, error)

    def get_info(self, *, name: str) -> Optional[LegacyDatasetInfo]:
        """Get metadata information for a registered dataset.
        
        Args:
            name: Name of the dataset.
            
        Returns:
            Dataset information if found, or None.
        """
        dataset: Optional[RegisteredDataset] = self.get(name=name)
        return dataset.info if dataset is not None else None

    def register_with_decorator(self, *, name: str, source: str, task_type: TaskType) -> Callable[[Type[Any]], Type[Any]]:
        """Decorator for registering a dataset class.
        
        Args:
            name: Name of the dataset.
            source: Source of the dataset.
            task_type: Type of task the dataset is for.
            
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
            if not hasattr(cls, 'info'):
                cls.info = LegacyDatasetInfo(name=name, source=source, task_type=task_type)
            self.register_new(name=name, dataset_cls=cls, info=cls.info)
            return cls

        return decorator

    def clear(self) -> None:
        """Clear all registered datasets from both new and legacy registries."""
        self._registry.clear()
        self._legacy_registry.clear()
        logger.debug("Cleared all registered datasets.")


# Global singleton for unified dataset registry
UNIFIED_REGISTRY: UnifiedDatasetRegistry = UnifiedDatasetRegistry()


# Decorator for registering datasets
def register(name: str, *, source: str, task_type: TaskType, description: str = "") -> Callable[[Type[Any]], Type[Any]]:
    """Decorator for registering a dataset class with the registry.
    
    Args:
        name: Name of the dataset.
        source: Source of the dataset.
        task_type: Type of task the dataset is for.
        description: Optional description of the dataset.
        
    Returns:
        Decorator function that registers the decorated class.
    """
    return UNIFIED_REGISTRY.register_with_decorator(
        name=name, source=source, task_type=task_type
    )


# Initialize the registry with core datasets
def initialize_registry() -> None:
    """Initialize the dataset registry with core datasets."""
    # Register core datasets from legacy registry
    from ember.core.utils.data.datasets_registry import (
        truthful_qa, mmlu, commonsense_qa, halueval, short_answer
    )

    # Register preppers from the legacy core registry
    UNIFIED_REGISTRY.register_metadata(
        name="truthful_qa",
        description="TruthfulQA dataset",
        source="truthful_qa",
        task_type=TaskType.MULTIPLE_CHOICE,
        prepper_class=truthful_qa.TruthfulQAPrepper
    )

    UNIFIED_REGISTRY.register_metadata(
        name="mmlu",
        description="Massive Multitask Language Understanding dataset",
        source="mmlu",
        task_type=TaskType.MULTIPLE_CHOICE,
        prepper_class=mmlu.MMLUPrepper
    )

    UNIFIED_REGISTRY.register_metadata(
        name="commonsense_qa",
        description="CommonsenseQA dataset",
        source="commonsense_qa",
        task_type=TaskType.MULTIPLE_CHOICE,
        prepper_class=commonsense_qa.CommonsenseQAPrepper
    )

    UNIFIED_REGISTRY.register_metadata(
        name="halueval",
        description="HaluEval dataset",
        source="halueval",
        task_type=TaskType.MULTIPLE_CHOICE,
        prepper_class=halueval.HaluEvalPrepper
    )

    UNIFIED_REGISTRY.register_metadata(
        name="my_shortanswer_ds",
        description="Short Answer dataset",
        source="short_answer",
        task_type=TaskType.SHORT_ANSWER,
        prepper_class=short_answer.ShortAnswerPrepper
    )

    # Discover datasets in the ember.data.datasets package
    UNIFIED_REGISTRY.discover_datasets() 