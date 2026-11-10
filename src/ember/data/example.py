"""Dataset records and dataset descriptions.

An `Example` is the one record type ember's datasets produce. A `Dataset`
names a dataset on the Hugging Face Hub and says, with a function the caller
can read, how each raw row becomes an `Example`. Nothing is inferred from
field names.
"""

import dataclasses
import string
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TypeAlias

__all__ = ["LETTERS", "Dataset", "Example", "Row", "Source"]

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"""Choice labels: the first choice is ``A``."""

Row: TypeAlias = Mapping[str, object]
"""One raw row of a dataset: each column's name and value."""

_PARQUET_FIELDS = frozenset({"config", "split"})


def _no_metadata() -> Mapping[str, str]:
    return MappingProxyType({})


@dataclasses.dataclass(frozen=True, slots=True)
class Example:
    """One question with its reference answer.

    Attributes:
        question: The question text.
        answer: The reference answer. For a multiple-choice question it is the
            letter of the correct choice, ``"A"`` for the first.
        choices: The options of a multiple-choice question, in order; empty
            for other questions.
        metadata: Other fields of the row worth keeping, such as MMLU's subject.
            The example keeps its own copy; it is not part of the hash.
    """

    question: str
    answer: str
    choices: tuple[str, ...] = ()
    metadata: Mapping[str, str] = dataclasses.field(default_factory=_no_metadata, hash=False)

    def __post_init__(self) -> None:
        self._check_types()
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if not self.question.strip():
            raise ValueError("an example needs a non-empty question")
        if not self.answer.strip():
            raise ValueError("an example needs a non-empty answer")
        if not self.choices:
            return
        count = len(self.choices)
        if not 2 <= count <= len(LETTERS):
            raise ValueError(f"a multiple-choice example needs 2 to 26 choices, not {count}")
        labels = LETTERS[:count]
        if len(self.answer) != 1 or self.answer not in labels:
            raise ValueError(f"the answer must be one of {', '.join(labels)}, not {self.answer!r}")

    def _check_types(self) -> None:
        for name, value in (("question", self.question), ("answer", self.answer)):
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string, got {type(value).__name__}")
        if not isinstance(self.choices, tuple) or not all(
            isinstance(choice, str) for choice in self.choices
        ):
            raise ValueError("choices must be a tuple of strings")
        if not isinstance(self.metadata, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in self.metadata.items()
        ):
            raise ValueError("metadata must map strings to strings")


@dataclasses.dataclass(frozen=True, slots=True)
class Dataset:
    """A dataset on the Hugging Face Hub and how its rows become examples.

    Attributes:
        path: Hub repository id, for example ``"openai/gsm8k"``.
        config: Hub configuration name, or ``None`` for the default one.
        split: The split `ember.data.stream` reads when given none.
        to_example: Builds an `Example` from one row. Raises `ValueError` when
            the row does not have the fields and values it expects.
        revision: Hub revision (commit, tag or branch) to read; ``None`` reads
            the default branch, which can change.
        parquet_file: Where in the repository each split is stored as one
            Parquet file, with ``{split}`` and optionally ``{config}`` standing
            for their values, for example
            ``"{config}/{split}-00000-of-00001.parquet"``. When set,
            `ember.data.stream` downloads that file and reads it from disk,
            and ``to_example`` gets each row's values as stored in the file
            (a column ``datasets`` would decode, such as an image, stays raw).
            When ``None``, it streams the split through the ``datasets``
            library.
    """

    path: str
    config: str | None
    split: str
    to_example: Callable[[Row], Example]
    revision: str | None = None
    parquet_file: str | None = None

    def __post_init__(self) -> None:
        for name, value in (("path", self.path), ("split", self.split)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")
        optionals = (
            ("config", self.config),
            ("revision", self.revision),
            ("parquet_file", self.parquet_file),
        )
        for name, optional in optionals:
            if optional is not None and (not isinstance(optional, str) or not optional):
                raise ValueError(f"{name} must be None or a non-empty string, got {optional!r}")
        if self.parquet_file is not None:
            _check_parquet_file(self.parquet_file, self.config)


def _check_parquet_file(template: str, config: str | None) -> None:
    try:
        parts = [
            (field, spec, conversion)
            for _, field, spec, conversion in string.Formatter().parse(template)
            if field is not None
        ]
    except ValueError as exc:
        raise ValueError(f"parquet_file {template!r} is not a valid template: {exc}") from None
    if any(spec or conversion for _, spec, conversion in parts):
        raise ValueError(f"parquet_file {template!r} may not convert or format its values")
    fields = {field for field, _, _ in parts}
    unknown = fields - _PARQUET_FIELDS
    if unknown:
        raise ValueError(
            f"parquet_file may use only {{config}} and {{split}}, not {sorted(unknown)}"
        )
    if "split" not in fields:
        raise ValueError("parquet_file must contain {split}")
    if "config" in fields and config is None:
        raise ValueError("parquet_file uses {config}, but config is None")


@dataclasses.dataclass(frozen=True, slots=True)
class Source:
    """Where a stream's examples come from; evaluation runs record it.

    Attributes:
        path: Hub repository id.
        config: Hub configuration name, or ``None`` for the default one.
        split: The split read.
        revision: The Hub revision read, or ``None`` for the default branch.
        limit: The most examples the stream yields, or ``None`` for all.
    """

    path: str
    config: str | None
    split: str
    revision: str | None
    limit: int | None
