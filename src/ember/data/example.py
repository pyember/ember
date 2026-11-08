"""Dataset records and dataset descriptions.

An `Example` is the one record type ember's datasets produce. A `Dataset`
names a dataset on the Hugging Face Hub and says, with a function the caller
can read, how each raw row becomes an `Example`. Nothing is inferred from
field names.
"""

import dataclasses
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TypeAlias

__all__ = ["LETTERS", "Dataset", "Example", "Row"]

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"""Choice labels: the first choice is ``A``."""

Row: TypeAlias = Mapping[str, object]
"""One raw row, as the Hugging Face ``datasets`` library yields it."""


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
    """

    question: str
    answer: str
    choices: tuple[str, ...] = ()
    metadata: Mapping[str, str] = dataclasses.field(default_factory=_no_metadata)

    def __post_init__(self) -> None:
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
    """

    path: str
    config: str | None
    split: str
    to_example: Callable[[Row], Example]
    revision: str | None = None
