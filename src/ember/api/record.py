"""Typed dataset record abstractions for Ember's streaming API."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

JSONScalar = Union[str, int, float, bool, None]
JSONValue = Union[JSONScalar, Sequence["JSONValue"], Mapping[str, "JSONValue"]]


def _coerce_json(value: Any, *, path: str) -> JSONValue:
    """Return ``value`` as a JSON-serializable structure or raise ``TypeError``."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, Mapping):
        coerced_items: Dict[str, JSONValue] = {}
        for key, inner in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"JSON object keys must be strings; got {type(key).__name__} at {path}"
                )
            coerced_items[key] = _coerce_json(inner, path=f"{path}.{key}")
        return MappingProxyType(coerced_items)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_coerce_json(item, path=f"{path}[]") for item in value)

    raise TypeError(f"Unsupported JSON value type {type(value).__name__} at {path}")


def _freeze_json(value: JSONValue) -> Any:
    """Return a hashable representation of ``value``."""
    if isinstance(value, Mapping):
        items = ((key, _freeze_json(inner)) for key, inner in sorted(value.items()))
        return tuple(items)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)

    return value


def _thaw_json(value: JSONValue) -> Any:
    """Convert JSONValue back to plain Python containers."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw_json(item) for item in value]
    return value


def _mapping_proxy(data: Optional[Mapping[str, JSONValue]]) -> Mapping[str, JSONValue]:
    if not data:
        return MappingProxyType({})
    if isinstance(data, MappingProxyType):
        return data
    coerced = {key: _coerce_json(value, path=f"metadata.{key}") for key, value in data.items()}
    return MappingProxyType(coerced)


@dataclass(frozen=True)
class TextContent:
    """Text payload with optional template metadata."""

    text: str
    template_id: Optional[str] = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - trivial setters
        object.__setattr__(self, "metadata", _mapping_proxy(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"text": self.text}
        if self.template_id is not None:
            data["template_id"] = self.template_id
        if self.metadata:
            data["metadata"] = _thaw_json(self.metadata)
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TextContent":
        return cls(
            text=str(payload.get("text", "")),
            template_id=payload.get("template_id"),
            metadata=payload.get("metadata", {}),
        )

    def __hash__(self) -> int:
        return hash((self.text, self.template_id, _freeze_json(self.metadata)))


@dataclass(frozen=True)
class Choice:
    """Single multiple-choice answer option."""

    label: str
    value: str
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - trivial setters
        object.__setattr__(self, "metadata", _mapping_proxy(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"label": self.label, "value": self.value}
        if self.metadata:
            data["metadata"] = _thaw_json(self.metadata)
        return data

    def __hash__(self) -> int:
        return hash((self.label, self.value, _freeze_json(self.metadata)))


@dataclass(frozen=True)
class ChoiceSet(Sequence[Choice]):
    """Ordered, immutable collection of choices."""

    _choices: Tuple[Choice, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:  # pragma: no cover - trivial setters
        object.__setattr__(self, "_choices", tuple(self._choices))

    def __len__(self) -> int:
        return len(self._choices)

    @overload
    def __getitem__(self, index: int) -> Choice: ...

    @overload
    def __getitem__(self, index: slice) -> Tuple[Choice, ...]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Choice, Tuple[Choice, ...]]:
        return self._choices[index]

    def __iter__(self) -> Iterator[Choice]:
        return iter(self._choices)

    def to_dict(self) -> Dict[str, Any]:
        return {choice.label: choice.value for choice in self._choices}

    def to_list(self) -> Sequence[Dict[str, Any]]:
        return [choice.to_dict() for choice in self._choices]

    @classmethod
    def empty(cls) -> "ChoiceSet":
        return cls(())

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ChoiceSet":
        sorted_items = sorted(mapping.items())
        return cls(
            tuple(Choice(label=str(label), value=str(value)) for label, value in sorted_items)
        )


@dataclass(frozen=True)
class MediaAsset:
    """Descriptor for a media asset referenced by a record."""

    uri: str
    kind: str
    sha256: Optional[str] = None
    format: Optional[str] = None
    dimensions: Optional[Tuple[int, ...]] = None
    content_type: Optional[str] = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    _opener: Optional[Callable[[], Any]] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:  # pragma: no cover - trivial setters
        dims: Optional[Tuple[int, ...]]
        if self.dimensions is None:
            dims = None
        else:
            dims = tuple(int(d) for d in self.dimensions)
        object.__setattr__(self, "dimensions", dims)
        object.__setattr__(self, "metadata", _mapping_proxy(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"uri": self.uri, "kind": self.kind}
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        if self.format is not None:
            data["format"] = self.format
        if self.dimensions is not None:
            data["dimensions"] = list(self.dimensions)
        if self.content_type is not None:
            data["content_type"] = self.content_type
        if self.metadata:
            data["metadata"] = _thaw_json(self.metadata)
        return data

    def with_opener(self, opener: Callable[[], Any]) -> "MediaAsset":
        return replace(self, _opener=opener)

    def open(self) -> Any:
        if self._opener is None:
            raise RuntimeError("MediaAsset has no opener; provide a cache or loader")
        return self._opener()

    def __hash__(self) -> int:
        components = (
            self.uri,
            self.kind,
            self.sha256,
            self.format,
            self.dimensions,
            self.content_type,
            _freeze_json(self.metadata),
        )
        return hash(components)


@dataclass(frozen=True)
class MediaBundle(Sequence[MediaAsset]):
    """Immutable sequence of media assets."""

    _assets: Tuple[MediaAsset, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:  # pragma: no cover - trivial setters
        object.__setattr__(self, "_assets", tuple(self._assets))

    def __len__(self) -> int:
        return len(self._assets)

    @overload
    def __getitem__(self, index: int) -> MediaAsset: ...

    @overload
    def __getitem__(self, index: slice) -> Tuple[MediaAsset, ...]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[MediaAsset, Tuple[MediaAsset, ...]]:
        return self._assets[index]

    def __iter__(self) -> Iterator[MediaAsset]:
        return iter(self._assets)

    def filter(self, *, kind: Optional[str] = None) -> "MediaBundle":
        if kind is None:
            return self
        return MediaBundle(tuple(asset for asset in self._assets if asset.kind == kind))

    def to_list(self) -> Sequence[Dict[str, Any]]:
        return [asset.to_dict() for asset in self._assets]

    @classmethod
    def empty(cls) -> "MediaBundle":
        return cls(())


@dataclass(frozen=True)
class DatasetRef:
    """Provenance information for a dataset record."""

    name: str
    subset: Optional[str] = None
    split: Optional[str] = None
    version: Optional[str] = None
    provider: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"name": self.name}
        if self.subset is not None:
            data["subset"] = self.subset
        if self.split is not None:
            data["split"] = self.split
        if self.version is not None:
            data["version"] = self.version
        if self.provider is not None:
            data["provider"] = self.provider
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetRef":
        return cls(
            name=str(payload["name"]),
            subset=payload.get("subset"),
            split=payload.get("split"),
            version=payload.get("version"),
            provider=payload.get("provider"),
        )


@dataclass(frozen=True)
class DataRecord:
    """Canonical Ember dataset record."""

    question: TextContent
    answer: TextContent
    choices: ChoiceSet = field(default_factory=ChoiceSet.empty)
    media: MediaBundle = field(default_factory=MediaBundle.empty)
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    source: Optional[DatasetRef] = None
    record_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _mapping_proxy(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "question": self.question.to_dict(),
            "answer": self.answer.to_dict(),
            "choices": self.choices.to_list(),
            "media": self.media.to_list(),
            "metadata": _thaw_json(self.metadata),
        }
        if self.source is not None:
            payload["source"] = self.source.to_dict()
        if self.record_id is not None:
            payload["record_id"] = self.record_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DataRecord":
        question = TextContent.from_dict(payload.get("question", {}))
        answer = TextContent.from_dict(payload.get("answer", {}))

        raw_choices = payload.get("choices")
        if isinstance(raw_choices, Mapping):
            choices = ChoiceSet.from_mapping(raw_choices)
        elif isinstance(raw_choices, Sequence):
            choices = ChoiceSet(
                tuple(
                    Choice(
                        label=str(item.get("label")),
                        value=str(item.get("value", "")),
                        metadata=item.get("metadata", {}),
                    )
                    for item in raw_choices
                )
            )
        else:
            choices = ChoiceSet.empty()

        raw_media = payload.get("media", [])
        if isinstance(raw_media, Sequence):
            media_assets: Tuple[MediaAsset, ...] = tuple(
                MediaAsset(
                    uri=str(item.get("uri")),
                    kind=str(item.get("kind")),
                    sha256=item.get("sha256"),
                    format=item.get("format"),
                    dimensions=(
                        tuple(item.get("dimensions", []))
                        if item.get("dimensions") is not None
                        else None
                    ),
                    content_type=item.get("content_type"),
                    metadata=item.get("metadata", {}),
                )
                for item in raw_media
            )
        else:
            media_assets = ()

        source_payload = payload.get("source")
        source = (
            DatasetRef.from_dict(source_payload) if isinstance(source_payload, Mapping) else None
        )

        metadata_payload = payload.get("metadata", {})
        if metadata_payload and not isinstance(metadata_payload, Mapping):
            raise TypeError("DataRecord metadata must be a mapping")
        return cls(
            question=question,
            answer=answer,
            choices=choices,
            media=MediaBundle(media_assets),
            metadata=metadata_payload,
            source=source,
            record_id=payload.get("record_id"),
        )

    def replace(self, **changes: Any) -> "DataRecord":
        return replace(self, **changes)

    def __hash__(self) -> int:  # pragma: no cover - simple wrapper
        components = (
            self.question,
            self.answer,
            self.choices,
            self.media,
            self.source,
            self.record_id,
            _freeze_json(self.metadata),
        )
        return hash(components)


__all__ = [
    "Choice",
    "ChoiceSet",
    "DataRecord",
    "DatasetRef",
    "MediaAsset",
    "MediaBundle",
    "TextContent",
]
