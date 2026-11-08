"""Streaming examples from the Hugging Face Hub."""

import itertools
from collections.abc import Iterable, Iterator, Mapping

from ember.data.example import Dataset, Example
from ember.data.named import DATASETS

__all__ = ["stream"]


def stream(
    dataset: str | Dataset, *, split: str | None = None, limit: int | None = None
) -> Iterator[Example]:
    """Streams a dataset's examples, in the dataset's order.

    Needs the ``data`` extra (``pip install 'ember-ai[data]'``). The dataset is
    opened when this is called, so a missing extra, an unknown dataset or an
    unknown split fails here; rows are then read as the iterator advances.
    Ember never runs code from the Hub (no ``trust_remote_code``).

    Args:
        dataset: A name in `ember.data.DATASETS`, or a `Dataset` describing any
            Hub dataset and how its rows become examples.
        split: The split to read; defaults to the dataset's own default.
        limit: Stop after this many examples.

    Returns:
        An iterator over the examples. It raises `ValueError` naming the row
        when a row does not fit the dataset's mapping.

    Raises:
        ValueError: ``dataset`` is not a known name, or ``limit`` is below 1.
        ImportError: The ``data`` extra is not installed.
    """
    spec = _resolve(dataset)
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    split = split or spec.split
    rows = _open(spec, split)
    return _examples(spec, split, itertools.islice(rows, limit))


def _resolve(dataset: str | Dataset) -> Dataset:
    if isinstance(dataset, Dataset):
        return dataset
    try:
        return DATASETS[dataset]
    except KeyError:
        known = ", ".join(sorted(DATASETS))
        raise ValueError(
            f"unknown dataset {dataset!r}; known: {known}. "
            "Pass an ember.data.Dataset to stream any Hugging Face dataset."
        ) from None


def _open(spec: Dataset, split: str) -> Iterable[object]:
    try:
        import datasets
    except ImportError as exc:
        raise ImportError("ember.data needs the data extra: pip install 'ember-ai[data]'") from exc
    rows: Iterable[object] = datasets.load_dataset(
        spec.path, spec.config, split=split, streaming=True, revision=spec.revision
    )
    return rows


def _examples(spec: Dataset, split: str, rows: Iterable[object]) -> Iterator[Example]:
    for index, row in enumerate(rows):
        where = f"{spec.path} ({spec.config or 'default'}, {split}) row {index}"
        if not isinstance(row, Mapping):
            raise ValueError(f"{where}: expected a mapping, got {type(row).__name__}")
        try:
            example = spec.to_example(row)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
        yield example
