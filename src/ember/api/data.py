"""Stream and load dataset records with a consistent schema.

The module exposes the Ember dataset API, including primitive data sources,
helpers for HuggingFace and local files, and a composable streaming iterator.
It favors lazy evaluation to minimize memory pressure while still surfacing
convenience helpers for eager loading.

Media Field Conventions:
    Input sources may use any of these field names for media attachments:
    - "media": Preferred field name for media asset descriptors
    - "images": Legacy alias, mapped to "media"
    - "query_images": LLMSELECTOR compatibility alias, mapped to "media"

    Media descriptors are dicts or lists of dicts with structure:
        {"uri": str, "kind": str, "sha256": str?, "dimensions": tuple?, ...}

    Output normalization:
    - normalize="record" (default): Yields DataRecord with typed .media field
    - normalize="dict": Yields dict with "media" list
    - normalize="none": Raw passthrough from source

Examples:
    >>> from ember.api.data import register, stream
    >>> class APISource:
    ...     def read_batches(self, batch_size: int = 32):
    ...         yield [{"question": "Q?", "answer": "A!"}]
    >>> register("tiny", APISource())
    >>> record = next(iter(stream("tiny")))
    >>> record.question.text
    'Q?'
"""

from __future__ import annotations

import csv
import json
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    cast,
    runtime_checkable,
)

from ember.api.record import ChoiceSet, DataRecord, MediaAsset, MediaBundle, TextContent

if TYPE_CHECKING:  # pragma: no cover
    from ember.api.record import MediaAsset


@runtime_checkable
class DataSource(Protocol):
    """Protocol for objects that stream dataset batches.

    A data source yields homogeneous dictionaries representing dataset rows.
    Custom implementations can wrap APIs, databases, or in-memory buffers so
    long as they expose :meth:`read_batches`.

    Examples:
        >>> class MemorySource:
        ...     def __init__(self, rows: list[dict[str, Any]]):
        ...         self._rows = rows
        ...
        ...     def read_batches(self, batch_size: int = 32):
        ...         for start in range(0, len(self._rows), batch_size):
        ...             yield self._rows[start : start + batch_size]
        >>> source = MemorySource([{\'value\': 1}, {\'value\': 2}])
        >>> next(source.read_batches())
        [{'value': 1}, {'value': 2}]
    """

    def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
        """Yield batches from the data source.

        Args:
            batch_size: Maximum number of records to include per batch.

        Yields:
            list[dict[str, Any]]: Batches of dataset records. The last batch may
            contain fewer than ``batch_size`` entries.
        """
        ...


@dataclass(frozen=True)
class DatasetInfo:
    """Container for dataset metadata used by the registry.

    Attributes:
        name: Registry identifier for the dataset.
        description: Human-readable summary of the dataset.
        size_bytes: Total size on disk. ``0`` indicates the size is unknown.
        example_count: Number of rows if known, otherwise ``0``.
        example_item: Representative record for documentation purposes.
        streaming_supported: ``True`` if the dataset supports streaming access.
    """

    name: str
    description: str
    size_bytes: int
    example_count: int
    example_item: Dict[str, Any]
    streaming_supported: bool = True


class _Operation(NamedTuple):
    """Internal representation of a pipeline step."""

    kind: Literal["filter", "transform"]
    func: Callable[[Any], Any]


NormalizeMode = Literal["record", "dict", "none"]
NormalizeInput = Union[NormalizeMode, bool]


def _coerce_normalize_mode(value: NormalizeInput) -> NormalizeMode:
    """Translate user input into the canonical normalization mode."""

    if value is True:
        return "dict"
    if value is False:
        return "none"
    if isinstance(value, str) and value in {"record", "dict", "none"}:
        return cast(NormalizeMode, value)
    raise ValueError(
        f"normalize must be one of {{'record', 'dict', 'none', True, False}}; got {value!r}"
    )


@runtime_checkable
class MediaCache(Protocol):
    """Protocol describing the media cache contract used during streaming."""

    def bind(self, asset: "MediaAsset") -> "MediaAsset":
        """Return an asset augmented with cache-aware loading semantics."""


@runtime_checkable
class SeedableSource(Protocol):
    """Protocol for sources that expose deterministic seeding hooks."""

    def with_seed(self, seed: Optional[int]) -> DataSource:
        """Return a source configured with ``seed``."""


def stream(
    source: Union[str, DataSource],
    *,
    subset: Optional[str] = None,
    split: Optional[str] = None,
    filter: Optional[Callable[[Any], bool]] = None,
    transform: Optional[Callable[[Any], Any]] = None,
    batch_size: int = 32,
    max_items: Optional[int] = None,
    normalize: NormalizeInput = "record",
    seed: Optional[int] = None,
    asset_cache: Optional[MediaCache] = None,
) -> StreamIterator:
    """Create a streaming iterator with optional filtering and transforms.

    The helper resolves the provided ``source`` to a :class:`DataSource`, then
    returns a :class:`StreamIterator` that supports chaining fluent operations
    such as ``filter`` and ``transform``.

    Args:
        source: Registered dataset name (for example, ``"mmlu"``) or a custom
            :class:`DataSource` implementation.
        subset: Dataset configuration or sub-split identifier understood by the
            backing source.
        split: Particular dataset split to read (for example ``"train"`` or
            ``"validation"``).
        filter: Callable that returns ``True`` for rows that should be kept.
        transform: Callable that receives each row and returns a transformed
            row.
        batch_size: Maximum item count requested from the source at once. This
            affects throughput but not the iterator interface.
        max_items: Hard cap on the number of records produced by the iterator.
        normalize: Output representation for items. ``"record"`` yields
            :class:`DataRecord` (default). ``"dict"`` produces the legacy
            normalized dictionary. ``"none"`` returns raw source payloads.
            ``True`` and ``False`` are accepted for backward compatibility and
            map to ``"dict"`` and ``"none"`` respectively.
        seed: Deterministic seed forwarded to sources that support sampling.
        asset_cache: Optional cache used to attach lazy openers to media assets.

    Returns:
        StreamIterator: Lazy iterator over dataset records that supports method
        chaining.

    Raises:
        ImportError: The HuggingFace ``datasets`` package is not installed while
            attempting to load a Hub dataset.
        FileNotFoundError: A :class:`FileSource` refers to a missing path.
        ValueError: A :class:`FileSource` receives an unsupported file type or
            configuration.

    Examples:
        >>> iterator = stream("mmlu")
        >>> first = next(iter(iterator))
        >>> isinstance(first, DataRecord)
        True
        >>> hard_physics = (
        ...     stream("mmlu", subset="physics")
        ...     .filter(lambda item: item.metadata.get("difficulty", 0) >= 4)
        ...     .limit(2)
        ... )
        >>> len(hard_physics.first(2))
        2
    """
    # Resolve registered names via the global registry so HuggingFace fallbacks work.
    if isinstance(source, str):
        data_source = _registry.get_source(source, subset, split, seed)
    else:
        data_source = source
        if seed is not None and isinstance(data_source, SeedableSource):
            data_source = data_source.with_seed(seed)

    normalize_mode = _coerce_normalize_mode(normalize)

    return StreamIterator(
        source=data_source,
        filter=filter,
        transform=transform,
        batch_size=batch_size,
        max_items=max_items,
        normalize=normalize_mode,
        seed=seed,
        asset_cache=asset_cache,
    )


class StreamIterator:
    """Composable iterator returned by :func:`stream`."""

    def __init__(
        self,
        source: DataSource,
        *,
        filter: Optional[Callable[[Any], bool]] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        batch_size: int = 32,
        max_items: Optional[int] = None,
        normalize: NormalizeMode = "record",
        seed: Optional[int] = None,
        asset_cache: Optional[MediaCache] = None,
        _operations: Optional[Tuple[_Operation, ...]] = None,
        trust_remote_code: bool = False,
        load_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Configure a new iterator instance."""
        self._source = source
        self._batch_size = batch_size
        self._max_items = max_items
        self._normalize_mode = normalize
        self._seed = seed
        self._asset_cache = asset_cache

        operations = list(_operations or ())
        if filter is not None:
            operations.append(_Operation("filter", filter))
        if transform is not None:
            operations.append(_Operation("transform", transform))
        self._operations: Tuple[_Operation, ...] = tuple(operations)

    def __iter__(self) -> Iterator[Any]:
        """Yield rows after applying normalization and the configured pipeline."""

        if self._max_items is not None and self._max_items <= 0:
            return

        count = 0
        for batch in self._source.read_batches(self._batch_size):
            for raw_item in batch:
                item = self._normalize_item(raw_item)

                for operation in self._operations:
                    if operation.kind == "filter":
                        if not operation.func(item):
                            break
                    else:
                        item = operation.func(item)
                else:
                    expected = self._expected_output_type()
                    if expected is not None and not isinstance(item, expected):
                        raise TypeError(
                            "Transform functions must return "
                            f"{expected.__name__}; got {type(item).__name__}"
                        )

                    yield item
                    count += 1
                    if self._max_items is not None and count >= self._max_items:
                        return

    def _expected_output_type(self) -> Optional[type]:
        if self._normalize_mode == "record":
            return DataRecord
        if self._normalize_mode in {"dict", "none"}:
            return dict
        return None

    def _normalize_item(self, raw_item: Dict[str, Any]) -> Any:
        if self._normalize_mode == "none":
            return raw_item
        if self._normalize_mode == "dict":
            return _normalize_dict(raw_item)

        record = _normalize_record(raw_item)
        if self._asset_cache is not None and len(record.media) > 0:
            record = _attach_media_cache(record, self._asset_cache)
        return record

    def records(self) -> "StreamIterator":
        """Return an iterator that yields :class:`DataRecord` instances."""
        if self._normalize_mode == "record":
            return self
        return StreamIterator(
            source=self._source,
            batch_size=self._batch_size,
            max_items=self._max_items,
            normalize="record",
            seed=self._seed,
            asset_cache=self._asset_cache,
            _operations=self._operations,
        )

    def as_dicts(self) -> "StreamIterator":
        """Return an iterator that yields legacy normalized dictionaries."""
        if self._normalize_mode == "dict":
            return self
        return StreamIterator(
            source=self._source,
            batch_size=self._batch_size,
            max_items=self._max_items,
            normalize="dict",
            seed=self._seed,
            asset_cache=self._asset_cache,
            _operations=self._operations,
        )

    def filter(self, predicate: Callable[[Any], bool]) -> "StreamIterator":
        """Return a new iterator that keeps rows matching ``predicate``."""
        return StreamIterator(
            source=self._source,
            batch_size=self._batch_size,
            max_items=self._max_items,
            normalize=self._normalize_mode,
            seed=self._seed,
            asset_cache=self._asset_cache,
            _operations=self._operations + (_Operation("filter", predicate),),
        )

    def transform(self, fn: Callable[[Any], Any]) -> "StreamIterator":
        """Return a new iterator that applies ``fn`` to each row."""
        return StreamIterator(
            source=self._source,
            batch_size=self._batch_size,
            max_items=self._max_items,
            normalize=self._normalize_mode,
            seed=self._seed,
            asset_cache=self._asset_cache,
            _operations=self._operations + (_Operation("transform", fn),),
        )

    def limit(self, n: int) -> "StreamIterator":
        """Return a new iterator capped to ``n`` emitted rows."""
        if n <= 0:
            raise ValueError("limit() requires n > 0")

        current_max = self._max_items
        new_max = n if current_max is None else min(current_max, n)

        return StreamIterator(
            source=self._source,
            batch_size=self._batch_size,
            max_items=new_max,
            normalize=self._normalize_mode,
            seed=self._seed,
            asset_cache=self._asset_cache,
            _operations=self._operations,
        )

    def first(self, n: int) -> List[Any]:
        """Return the first ``n`` rows as a list."""
        return list(self.limit(n))

    def collect(self) -> List[Any]:
        """Materialize all remaining rows into memory."""
        if self._max_items is None:
            warnings.warn(
                "Collecting without a limit may exhaust memory; call limit() first.",
                RuntimeWarning,
                stacklevel=2,
            )

        return list(self)


def load(
    source: Union[str, DataSource],
    *,
    subset: Optional[str] = None,
    split: Optional[str] = None,
    filter: Optional[Callable[[Any], bool]] = None,
    transform: Optional[Callable[[Any], Any]] = None,
    max_items: Optional[int] = None,
    normalize: NormalizeInput = "record",
    batch_size: int = 32,
    seed: Optional[int] = None,
    asset_cache: Optional[MediaCache] = None,
) -> List[Any]:
    """Eagerly materialize records from :func:`stream`.

    Args:
        source: Dataset identifier or :class:`DataSource` instance.
        subset: Dataset configuration identifier.
        split: Dataset split to read.
        filter: Predicate applied to each row before collection.
        transform: Callable applied to every row.
        max_items: Maximum number of rows to store in memory.
        normalize: Output representation for items (see :func:`stream`).
        batch_size: Number of rows to request from the underlying source.
        seed: Deterministic seed forwarded to sources as applicable.
        asset_cache: Optional cache used for media resolution.

    Returns:
        list[Any]: Materialized dataset rows.

    Raises:
        ImportError: Raised for missing HuggingFace dependencies when loading a
            Hub dataset.
        FileNotFoundError: Raised when a :class:`FileSource` path cannot be
            located.
        ValueError: Raised when a :class:`FileSource` receives an unsupported
            file format.

    Examples:
        >>> rows = load(
        ...     "mmlu",
        ...     subset="physics",
        ...     filter=lambda row: row["metadata"].get("difficulty") == "hard",
        ...     max_items=2,
        ... )
        >>> len(rows)
        2
    """
    return list(
        stream(
            source,
            subset=subset,
            split=split,
            filter=filter,
            transform=transform,
            max_items=max_items,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            asset_cache=asset_cache,
        )
    )


def metadata(dataset: str) -> DatasetInfo:
    """Return metadata associated with ``dataset``.

    Args:
        dataset: Dataset identifier previously registered or available on the
            HuggingFace Hub.

    Returns:
        DatasetInfo: Metadata describing the dataset. When the dataset cannot be
        loaded, a placeholder with default values is returned.

    Examples:
        >>> info = metadata("mmlu")
        >>> info.name
        'mmlu'
    """
    return _registry.get_metadata(dataset)


def list_datasets() -> List[str]:
    """Return the sorted list of datasets registered in the global registry.

    Returns:
        list[str]: Alphabetically sorted dataset identifiers currently registered.

    Note:
        The list reflects explicit registrations only. HuggingFace datasets that
        can be streamed implicitly will not appear here until registered.

    Examples:
        >>> isinstance(list_datasets(), list)
        True
    """
    return _registry.list_available()


def register(name: str, source: DataSource, metadata: Optional[DatasetInfo] = None) -> None:
    """Register a named data source for later lookup.

    Args:
        name: Unique identifier used with :func:`stream` and :func:`load`.
        source: Object implementing the :class:`DataSource` protocol.
        metadata: Optional descriptive information surfaced through
            :func:`metadata`. Provide this to avoid lazy inspection when
            documenting the dataset.

    Raises:
        TypeError: ``source`` does not satisfy the :class:`DataSource` protocol.

    Examples:
        >>> register("support_logs", FileSource("logs.jsonl"))
        >>> class RedisSource:
        ...     def read_batches(self, batch_size: int = 32):
        ...         yield from iter_redis_batches(batch_size)
        >>> register("tickets", RedisSource())
    """
    if not isinstance(source, DataSource):
        raise TypeError(f"Source must implement DataSource protocol. Got {type(source).__name__}")
    _registry.register(name, source, metadata)


def from_file(path: Union[str, Path], **kwargs: Any) -> StreamIterator:
    """Create a :class:`StreamIterator` from a JSON, JSONL, or CSV file.

    Args:
        path: File system path to load.
        **kwargs: Additional keyword arguments forwarded to :func:`stream`.

    Returns:
        StreamIterator: Lazy iterator over file rows.

    Raises:
        FileNotFoundError: The file does not exist.
        ValueError: The extension is unsupported.

    Examples:
        >>> rows = from_file(  # doctest: +SKIP
        ...     "scores.csv",
        ...     filter=lambda row: float(row["score"]) > 0.8,
        ... )
        >>> hasattr(rows, "first")  # doctest: +SKIP
        True
    """
    return stream(FileSource(path), **kwargs)


def load_file(path: Union[str, Path], **kwargs: Any) -> List[Dict[str, Any]]:
    """Return the contents of a structured data file as a list.

    Args:
        path: File system path to load.
        **kwargs: Arguments forwarded to :func:`load`.

    Returns:
        list[dict[str, Any]]: Materialized rows read from ``path``.

    Raises:
        FileNotFoundError: The file does not exist.
        ValueError: The extension is unsupported.

    Examples:
        >>> import json
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as tmp:
        ...     sample = Path(tmp) / "rows.json"
        ...     records = [{"question": "Life?", "answer": "42"}]
        ...     sample.write_text(json.dumps(records))
        ...     load_file(sample)[0]["answer"]
        '42'
    """
    return load(FileSource(path), **kwargs)


def _normalize_dict(item: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce dataset rows into Ember's canonical schema.

    Args:
        item: Arbitrary mapping produced by a data source.

    Returns:
        dict[str, Any]: Dictionary containing ``question``, ``answer``,
        ``choices``, and ``metadata`` keys.

    Examples:
        >>> _normalize({"prompt": "What is 2+2?", "response": "4"})
        {'question': 'What is 2+2?', 'answer': '4', 'choices': {}, 'metadata': {}}
    """

    def _first_present(keys: Tuple[str, ...], default: Any = "") -> Any:
        for key in keys:
            if key in item:
                value = item[key]
                if value is not None:
                    return value
        return default

    question_default = item.get("input")
    if question_default is None:
        question_default = ""
    question = _first_present(("question", "query", "prompt", "text"), question_default)

    answer_default = item.get("response")
    if answer_default is None:
        answer_default = ""
    answer = _first_present(("answer", "target", "label", "output"), answer_default)

    # Extract choices, converting list to dict if needed
    choices = item.get("choices") or item.get("options", {})
    if isinstance(choices, list):
        # Convert ["opt1", "opt2"] to {"A": "opt1", "B": "opt2"}
        choices = {chr(65 + i): choice for i, choice in enumerate(choices)}
    elif not isinstance(choices, dict):
        choices = {}

    # Build normalized item
    normalized = {
        "question": question,
        "answer": answer,
        "choices": choices,
        "metadata": {},
    }

    # Preserve media descriptors when present.
    raw_media = item.get("media")
    if isinstance(raw_media, list):
        normalized["media"] = raw_media
    elif isinstance(raw_media, dict):
        normalized["media"] = [raw_media]

    # Handle existing metadata field
    if "metadata" in item and isinstance(item["metadata"], dict):
        normalized["metadata"].update(item["metadata"])

    # Add remaining fields to metadata
    excluded_keys = {
        "question",
        "query",
        "prompt",
        "text",
        "input",
        "answer",
        "target",
        "label",
        "output",
        "response",
        "choices",
        "options",
        "metadata",
        "media",
    }

    for key, value in item.items():
        if key not in excluded_keys:
            normalized["metadata"][key] = value

    return normalized


def _coerce_media_bundle(raw_media: Any) -> MediaBundle:
    """Convert raw media descriptors into a :class:`MediaBundle`."""

    if raw_media is None:
        return MediaBundle.empty()

    if isinstance(raw_media, Mapping):
        media_iterable: Sequence[Any] = [raw_media]
    elif isinstance(raw_media, Sequence) and not isinstance(raw_media, (str, bytes, bytearray)):
        media_iterable = raw_media
    else:
        return MediaBundle.empty()

    assets: list[MediaAsset] = []
    for entry in media_iterable:
        if not isinstance(entry, Mapping):
            continue

        uri = entry.get("uri") or entry.get("path") or entry.get("url")
        if uri is None:
            continue

        kind_value = entry.get("kind") or entry.get("type") or entry.get("role") or "unknown"
        sha_value = entry.get("sha256")
        format_value = entry.get("format") or entry.get("file_format")
        content_type_value = entry.get("content_type") or entry.get("mime_type")
        dims_value = entry.get("dimensions")

        dims_iterable = (
            tuple(int(d) for d in dims_value)
            if isinstance(dims_value, Sequence)
            and not isinstance(dims_value, (str, bytes, bytearray))
            else None
        )

        metadata_payload = entry.get("metadata")
        metadata_dict: Dict[str, Any] = (
            dict(metadata_payload) if isinstance(metadata_payload, Mapping) else {}
        )

        metadata_dict.update(
            {
                key: value
                for key, value in entry.items()
                if key
                not in {
                    "uri",
                    "path",
                    "url",
                    "kind",
                    "type",
                    "role",
                    "sha256",
                    "format",
                    "file_format",
                    "dimensions",
                    "content_type",
                    "mime_type",
                    "metadata",
                }
            }
        )

        assets.append(
            MediaAsset(
                uri=str(uri),
                kind=str(kind_value),
                sha256=str(sha_value) if sha_value is not None else None,
                format=str(format_value) if format_value is not None else None,
                dimensions=dims_iterable,
                content_type=str(content_type_value) if content_type_value is not None else None,
                metadata=metadata_dict,
            )
        )

    if not assets:
        return MediaBundle.empty()
    return MediaBundle(tuple(assets))


def _normalize_record(item: Dict[str, Any]) -> DataRecord:
    """Build a :class:`DataRecord` from ``item`` using legacy normalization."""

    normalized = _normalize_dict(item)
    choices = (
        ChoiceSet.from_mapping(normalized["choices"])
        if normalized["choices"]
        else ChoiceSet.empty()
    )

    raw_media = item.get("media")
    if raw_media is None and "media" in normalized.get("metadata", {}):
        raw_media = normalized["metadata"].pop("media")

    media_bundle = _coerce_media_bundle(raw_media)

    return DataRecord(
        question=TextContent(text=str(normalized["question"])),
        answer=TextContent(text=str(normalized["answer"])),
        choices=choices,
        media=media_bundle,
        metadata=normalized["metadata"],
    )


def _attach_media_cache(record: DataRecord, cache: MediaCache) -> DataRecord:
    """Return ``record`` with assets rebound using ``cache``."""

    if len(record.media) == 0:
        return record

    rebound = tuple(cache.bind(asset) for asset in record.media)
    if all(
        new_asset is original for new_asset, original in zip(rebound, record.media, strict=False)
    ):
        return record
    return record.replace(media=MediaBundle(rebound))


def _normalize(item: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible alias for callers expecting dict normalization."""

    return _normalize_dict(item)


class HuggingFaceSource:
    """Data source backed by the HuggingFace Datasets library.

    Examples:
        >>> register("squad", HuggingFaceSource("squad", split="validation"))
    """

    def __init__(
        self,
        name: str,
        split: Optional[str] = None,
        config: Optional[str] = None,
        seed: Optional[int] = None,
        *,
        trust_remote_code: bool = False,
    ):
        """Create a source referencing a dataset on the HuggingFace Hub.

        Args:
            name: Dataset identifier on the Hub.
            split: Dataset split name. Defaults to ``"train"``.
            config: Optional configuration for multi-config datasets.
        """
        self.name = name
        self.split = split or "train"
        self.config = config
        self.seed = seed
        self.trust_remote_code = trust_remote_code

    def with_config(
        self, subset: Optional[str] = None, split: Optional[str] = None
    ) -> HuggingFaceSource:
        """Return a copy with updated configuration or split.

        Args:
            subset: Configuration override passed to the Hub loader.
            split: Split override.

        Returns:
            HuggingFaceSource: New instance with the requested adjustments.
        """
        return HuggingFaceSource(
            name=self.name,
            split=split or self.split,
            config=subset or self.config,
            seed=self.seed,
            trust_remote_code=self.trust_remote_code,
        )

    def with_seed(self, seed: Optional[int]) -> HuggingFaceSource:
        """Return a copy with deterministic seeding applied."""

        return HuggingFaceSource(
            name=self.name,
            split=self.split,
            config=self.config,
            seed=seed,
            trust_remote_code=self.trust_remote_code,
        )

    def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
        """Yield batches from the remote dataset.

        Args:
            batch_size: Maximum number of rows to include per batch.

        Yields:
            list[dict[str, Any]]: Dataset rows converted to dictionaries.

        Raises:
            ImportError: The ``datasets`` package is missing.
            FileNotFoundError: The dataset, configuration, or split is not
                available on the Hub.
            ValueError: The underlying loader rejects the provided arguments.
        """
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "HuggingFace datasets not installed. Install with: pip install datasets"
            ) from e

        load_kwargs = {
            "split": self.split,
            "streaming": True,
        }
        if self.trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        dataset = load_dataset(
            self.name,
            self.config,
            **load_kwargs,
        )

        if self.seed is not None:
            dataset = dataset.shuffle(seed=self.seed)

        # Yield batches from a fresh iterator each call
        batch = []
        for item in dataset:
            batch.append(dict(item))
            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield final partial batch
        if batch:
            yield batch


class FileSource:
    """Disk-backed :class:`DataSource` for JSON, JSONL, and CSV files.

    The file type is inferred from the extension.

    Examples:
        >>> source = FileSource("data/train.jsonl")  # doctest: +SKIP
        >>> first_batch = next(source.read_batches(batch_size=2))  # doctest: +SKIP
    """

    def __init__(self, path: Union[Path, str]):
        """Create a file-backed source.

        Args:
            path: Path to the structured data file.

        Raises:
            FileNotFoundError: The path does not exist.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

    def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
        """Yield batches parsed from ``path``.

        Args:
            batch_size: Maximum number of rows per batch.

        Yields:
            list[dict[str, Any]]: Dictionaries parsed from the file.

        Raises:
            ValueError: The extension is unsupported.
            json.JSONDecodeError: The file contains malformed JSON content.
        """
        suffix = self.path.suffix.lower()

        if suffix == ".jsonl":
            yield from self._read_jsonl(batch_size)
        elif suffix == ".json":
            yield from self._read_json(batch_size)
        elif suffix == ".csv":
            yield from self._read_csv(batch_size)
        else:
            raise ValueError(f"Unsupported file type: {suffix}\nSupported: .json, .jsonl, .csv")

    def _read_jsonl(self, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
        """Yield parsed rows from a JSON Lines file.

        Args:
            batch_size: Maximum number of rows per batch.

        Yields:
            list[dict[str, Any]]: Dictionaries decoded from each line.

        Raises:
            json.JSONDecodeError: A line cannot be decoded.
        """
        batch = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    batch.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON on line {line_num}: {e.msg}", e.doc, e.pos
                    ) from e

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch

    def _read_json(self, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
        """Yield rows parsed from a JSON document.

        Args:
            batch_size: Maximum number of elements per batch when the root is a
                list.

        Yields:
            list[dict[str, Any]]: Dictionaries parsed from the JSON content.

        Raises:
            ValueError: The JSON root is neither a list nor an object.
            json.JSONDecodeError: The file contains malformed JSON.
        """
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            # Yield batches from list
            for i in range(0, len(data), batch_size):
                yield data[i : i + batch_size]
        elif isinstance(data, dict):
            # Single object
            yield [data]
        else:
            raise ValueError(f"JSON file must contain array or object, got {type(data).__name__}")

    def _read_csv(self, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
        """Yield dictionaries from a CSV file using the header row as keys.

        Args:
            batch_size: Maximum number of rows per batch.

        Yields:
            list[dict[str, Any]]: Parsed rows.
        """
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            batch = []

            for row in reader:
                # Convert to regular dict to ensure JSON serializable
                batch.append(dict(row))

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch


class _RegistryClass:
    """Thread-safe registry for named data sources and metadata."""

    def __init__(self) -> None:
        """Initialize the registry and preload standard datasets."""
        self._sources: Dict[str, DataSource] = {}
        self._metadata: Dict[str, DatasetInfo] = {}
        self._lock = threading.Lock()
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Register built-in evaluation datasets."""
        # Standard evaluation datasets
        standard_datasets = {
            "mmlu": ("cais/mmlu", "test", "all"),
            "squad": ("squad", "validation", None),
            "gsm8k": ("gsm8k", "test", "main"),
            "hellaswag": ("Rowan/hellaswag", "validation", None),
            "truthfulqa": ("truthful_qa", "validation", "multiple_choice"),
            "arc": ("ai2_arc", "test", "ARC-Challenge"),
            "winogrande": ("winogrande", "validation", "winogrande_xl"),
        }

        for name, (hf_name, split, config) in standard_datasets.items():
            source = HuggingFaceSource(hf_name, split, config)
            self.register(name, source)

        self._register_optional_datasets()

    def _register_optional_datasets(self) -> None:
        """Register optional benchmark datasets without creating import cycles."""

        from ember.api import datasets as _datasets  # local import to avoid circular dependency

        _datasets.register_builtin_datasets(self.register)

    def register(
        self, name: str, source: DataSource, metadata: Optional[DatasetInfo] = None
    ) -> None:
        """Store a source and optional metadata under ``name``."""
        with self._lock:
            self._sources[name] = source
            if metadata is not None:
                self._metadata[name] = metadata
            else:
                self._metadata.pop(name, None)

    def get_source(
        self,
        name: str,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> DataSource:
        """Return a source by name, falling back to a HuggingFace loader."""
        with self._lock:
            if name in self._sources:
                source = self._sources[name]
                # Handle sources that support configuration
                if hasattr(source, "with_config"):
                    source = source.with_config(subset=subset, split=split)
                if seed is not None and isinstance(source, SeedableSource):
                    source = source.with_seed(seed)
                return source

        # Try as HuggingFace dataset if not in registry
        return HuggingFaceSource(name, split=split, config=subset, seed=seed)

    def get_metadata(self, name: str) -> DatasetInfo:
        """Return cached metadata or synthesize a placeholder when missing.

        Args:
            name: Dataset identifier being requested.

        Returns:
            DatasetInfo describing the dataset, populating defaults if the
            registry cannot load example data.
        """
        with self._lock:
            if name in self._metadata:
                return self._metadata[name]

        # Generate metadata by loading one example
        try:
            example = None
            for item in stream(name, max_items=1, normalize=False):
                example = item
                break

            metadata = DatasetInfo(
                name=name,
                description=f"Dataset: {name}",
                size_bytes=0,  # Unknown without full scan
                example_count=0,  # Unknown without full scan
                example_item=example or {},
                streaming_supported=True,
            )

            with self._lock:
                self._metadata[name] = metadata

            return metadata

        except Exception:
            # Return minimal metadata on error
            return DatasetInfo(
                name=name,
                description=f"Dataset: {name} (error loading)",
                size_bytes=0,
                example_count=0,
                example_item={},
                streaming_supported=True,
            )

    def list_available(self) -> List[str]:
        """Return all registered dataset names sorted alphabetically."""
        with self._lock:
            return sorted(list(self._sources.keys()))


# Global registry instance
_registry = _RegistryClass()


__all__ = [
    # Main functions
    "stream",
    "load",
    "metadata",
    "list_datasets",
    "register",
    # File convenience functions
    "from_file",
    "load_file",
    # Types and classes
    "DataSource",
    "DatasetInfo",
    "StreamIterator",
    "FileSource",
    "HuggingFaceSource",
]
