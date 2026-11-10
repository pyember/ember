"""Examples from the Hugging Face Hub, read from a downloaded file or streamed."""

import importlib
import itertools
from collections.abc import Iterable, Iterator, Mapping
from types import ModuleType

from ember.data.example import Dataset, Example, Source
from ember.data.named import DATASETS

__all__ = ["Stream", "stream"]

# The modules whose absence means the data extra is not installed. Any other
# ImportError raised while importing them is a broken install and propagates.
_EXTRA_MODULES = frozenset({"datasets", "huggingface_hub", "pyarrow"})

# Rows decoded from a Parquet file at a time, so a stream with a small limit
# decodes little more than it yields.
_BATCH_ROWS = 1024


class Stream(Iterator[Example]):
    """Examples read from one split of one dataset, in the dataset's order."""

    def __init__(self, source: Source, examples: Iterator[Example]) -> None:
        self._source = source
        self._examples = examples

    @property
    def source(self) -> Source:
        """Where the examples come from."""
        return self._source

    def __iter__(self) -> "Stream":
        return self

    def __next__(self) -> Example:
        return next(self._examples)


def stream(dataset: str | Dataset, *, split: str | None = None, limit: int | None = None) -> Stream:
    """Streams a dataset's examples, in the dataset's order.

    Needs the ``data`` extra (``pip install 'ember-ai[data]'``). The dataset is
    opened when this is called, so a missing extra, an unknown dataset or an
    unknown split fails here; rows are then read as the stream advances.
    Ember never runs code from the Hub (no ``trust_remote_code``).

    A dataset with a ``parquet_file``, such as every named dataset, is read
    from that file: the first call downloads the whole split's file into the
    Hugging Face cache, and later calls at the same pinned commit read it
    from there without the network. Any other dataset
    is streamed by the ``datasets`` library. On macOS with Python 3.13, a
    program that stops such a stream early (with ``limit``, say) and exits at
    once has been seen to hang at exit, held up by that library's reader.

    Args:
        dataset: A name in `ember.data.DATASETS`, or a `Dataset` describing any
            Hub dataset and how its rows become examples.
        split: The split to read; defaults to the dataset's own default.
        limit: Stop after this many examples.

    Returns:
        The examples, with their `Source`. Advancing raises `ValueError` naming
        the row when a row does not fit the dataset's mapping.

    Raises:
        ValueError: ``dataset`` is not a known name, ``split`` is empty, or
            ``limit`` is below 1.
        ImportError: The ``data`` extra is not installed.
        huggingface_hub errors: Raised unchanged, among them the Hub's error
            for a ``parquet_file`` that does not exist, which is how an
            unknown split of a named dataset fails.
    """
    spec = _resolve(dataset)
    if split is not None and (not isinstance(split, str) or not split):
        raise ValueError(f"split must be a non-empty string, got {split!r}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    split = spec.split if split is None else split
    rows = _open(spec, split)
    source = Source(spec.path, spec.config, split, spec.revision, limit)
    return Stream(source, _examples(spec, split, itertools.islice(rows, limit)))


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
    if spec.parquet_file is None:
        rows: Iterable[object] = _extra("datasets").load_dataset(
            spec.path, spec.config, split=split, streaming=True, revision=spec.revision
        )
        return rows
    # Both are imported before the download, so a missing extra never starts one.
    hub = _extra("huggingface_hub")
    parquet = _extra("pyarrow.parquet")
    local = hub.hf_hub_download(
        repo_id=spec.path,
        filename=spec.parquet_file.format(config=spec.config, split=split),
        repo_type="dataset",
        revision=spec.revision,
    )
    # Opened by path and decoded on this thread: an abandoned read then leaves no
    # Python object for Arrow's threads to release while the interpreter exits.
    # tests/test_data.py pins both.
    batches = parquet.ParquetFile(local).iter_batches(batch_size=_BATCH_ROWS, use_threads=False)
    return itertools.chain.from_iterable(batch.to_pylist() for batch in batches)


def _extra(module: str) -> ModuleType:
    """Imports ``module`` from the data extra, naming the install command if it is missing."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        # Only a package that is not there at all; any other ImportError, such
        # as a name missing from an installed package, is a broken install.
        if exc.name not in _EXTRA_MODULES:
            raise
        raise ImportError(
            "ember.data needs the data extra: pip install 'ember-ai[data]'", name=exc.name
        ) from exc


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
