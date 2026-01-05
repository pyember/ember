"""Dataset-specific registration helpers."""

from __future__ import annotations

import importlib
from typing import Callable, Iterable, Optional

from ember.api.data import DatasetInfo, DataSource
from ember.api.datasets.catalog import DatasetCatalogEntry, DatasetSubtask, list_catalog_entries

RegisterFn = Callable[[str, DataSource, Optional[DatasetInfo]], None]

_DATASET_MODULES: tuple[str, ...] = (
    "ember.api.datasets.simpleqa",
    "ember.api.datasets.fever",
    "ember.api.datasets.mathvista",
    "ember.api.datasets.livecodebench",
    "ember.api.datasets.agieval",
    "ember.api.datasets.bbeh",
    "ember.api.datasets.mmlu_pro",
    "ember.api.datasets.halueval",
    "ember.api.datasets.mbpp",
    "ember.api.datasets.table_tasks",
)


def register_builtin_datasets(register: RegisterFn) -> None:
    for module_name in _DATASET_MODULES:
        module = importlib.import_module(module_name)
        register_callable = getattr(module, "register", None)
        if register_callable is None:
            raise AttributeError(f"Dataset module '{module_name}' missing register() function")
        register_callable(register)


def iter_dataset_metadata() -> Iterable[DatasetInfo]:
    for module_name in _DATASET_MODULES:
        module = importlib.import_module(module_name)
        metadata = getattr(module, "METADATA", None)
        if metadata is None:
            continue
        if isinstance(metadata, DatasetInfo):
            yield metadata
        elif isinstance(metadata, Iterable):
            for item in metadata:
                if isinstance(item, DatasetInfo):
                    yield item


def list() -> tuple[DatasetCatalogEntry, ...]:
    """Return catalog entries describing available datasets and subtasks."""

    return tuple(list_catalog_entries())


__all__ = [
    "register_builtin_datasets",
    "iter_dataset_metadata",
    "list",
    "DatasetCatalogEntry",
    "DatasetSubtask",
]
