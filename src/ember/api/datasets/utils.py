"""Lightweight helpers for adapting dataset sources.

The utilities in this module keep dataset registrations concise: a small
wrapper composes filtering and record rewriting, and a path resolver locates
local dataset payloads following Ember's conventions.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ember.api.data import DataSource, SeedableSource
from ember.api.exceptions import DatasetNotFoundError

FilterFn = Callable[[dict[str, Any]], bool]
TransformFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(slots=True, frozen=True)
class TransformingSource:
    """Apply optional filtering and transformation to another data source."""

    base: DataSource
    filter_fn: FilterFn | None = None
    transform_fn: TransformFn | None = None

    def read_batches(self, batch_size: int = 32) -> Iterator[list[dict[str, Any]]]:
        filter_fn = self.filter_fn
        transform_fn = self.transform_fn

        for batch in self.base.read_batches(batch_size):
            if not filter_fn and not transform_fn:
                yield batch
                continue

            processed: list[dict[str, Any]] = []
            for item in batch:
                if filter_fn and not filter_fn(item):
                    continue
                if transform_fn:
                    item = transform_fn(item)
                    if item is None:
                        continue  # Skip items that transform_fn signals to drop
                processed.append(item)

            if processed:
                yield processed

    def with_seed(self, seed: int | None) -> "TransformingSource":
        if isinstance(self.base, SeedableSource):
            return TransformingSource(
                self.base.with_seed(seed),
                filter_fn=self.filter_fn,
                transform_fn=self.transform_fn,
            )
        return self


# Backward-compatible alias retained for datasets that emphasize rewriting.
DataTransformer = TransformingSource


def resolve_dataset_path(
    dataset: str,
    *,
    env_var: str | None = None,
    required_files: Sequence[str] | None = None,
) -> Path:
    """Return the directory or file path containing a dataset.

    The search order is:
        1. Explicit environment variable when provided.
        2. ``EMBER_DATASETS_ROOT``/<dataset>
        3. ``~/.ember/datasets``/<dataset>

    Args:
        dataset: Canonical dataset identifier.
        env_var: Optional environment variable that points directly to the
            dataset root (file or directory).
        required_files: Optional collection of files that must exist relative to
            the resolved path (only checked when the path is a directory).

    Raises:
        DatasetNotFoundError: When no matching path can be located.
    """

    search_paths: list[Path] = []
    if env_var and (env_value := os.environ.get(env_var)):
        search_paths.append(Path(env_value).expanduser())

    if datasets_root := os.environ.get("EMBER_DATASETS_ROOT"):
        search_paths.append(Path(datasets_root).expanduser() / dataset)

    search_paths.append(Path.home() / ".ember" / "datasets" / dataset)

    required = tuple(required_files or ())

    for candidate in search_paths:
        resolved = _resolve_candidate_path(candidate, required)
        if resolved is not None:
            return resolved

    raise DatasetNotFoundError.for_dataset(dataset)


def _all_files_present(root: Path, filenames: Iterable[str]) -> bool:
    return all((root / name).exists() for name in filenames)


def _resolve_candidate_path(candidate: Path, required: Sequence[str]) -> Path | None:
    candidate = candidate.expanduser()

    if candidate.is_file():
        return candidate

    if not candidate.is_dir():
        return None

    if required and not _all_files_present(candidate, required):
        return None

    return candidate
