"""Scorers compare a task's output with an example's reference answer.

Each built-in scorer applies one documented rule to read an answer out of the
output, and records what it read, so every score can be explained from the
output alone. None of them gives credit for an answer that merely appears
somewhere in the output.
"""

import dataclasses
import decimal
import re
from typing import Protocol, TypeVar

from ember.data import LETTERS, Example

__all__ = [
    "ExactMatch",
    "MultipleChoice",
    "NumericAnswer",
    "Score",
    "Scorer",
    "multiple_choice_prompt",
]

ItemT_contra = TypeVar("ItemT_contra", contravariant=True)
OutputT_contra = TypeVar("OutputT_contra", contravariant=True)

# A number written with digits: an optional minus sign, optional comma
# thousands separators, and an optional decimal part.
_NUMBER = re.compile(r"(?<![\d.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\d])")


@dataclasses.dataclass(frozen=True, slots=True)
class Score:
    """How one output scored.

    Attributes:
        value: 1.0 for a correct answer and 0.0 for a wrong one; scorers that
            give partial credit use values in between.
        extracted: The answer the scorer read from the output, or ``None``
            when the output contained none.
    """

    value: float
    extracted: str | None

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"a score must be in [0, 1], got {self.value}")


class Scorer(Protocol[ItemT_contra, OutputT_contra]):
    """Scores one output against the item it answers.

    A scorer returns a `Score` for every output, including wrong or empty
    ones; it raises only when the item itself cannot be scored this way.
    """

    def __call__(self, item: ItemT_contra, output: OutputT_contra, /) -> Score: ...


@dataclasses.dataclass(frozen=True, slots=True)
class ExactMatch:
    """Correct when the output equals the answer, ignoring surrounding whitespace.

    Attributes:
        case_sensitive: Whether letter case must match too.
    """

    case_sensitive: bool = True

    def __call__(self, example: Example, output: str, /) -> Score:
        got, want = output.strip(), example.answer.strip()
        if not self.case_sensitive:
            got, want = got.casefold(), want.casefold()
        return Score(1.0 if got == want else 0.0, output.strip())


@dataclasses.dataclass(frozen=True, slots=True)
class NumericAnswer:
    """Correct when the output's final number equals the numeric answer.

    Numbers are written with digits, an optional minus sign, optional comma
    thousands separators and an optional decimal part; ``1,000``, ``1000`` and
    ``1000.0`` are equal. Without ``after`` the scorer reads the last number in
    the output. With ``after`` it reads the first number following the last
    occurrence of that marker, and an output without the marker has no answer.

    Attributes:
        after: A marker the answer follows, for example ``"####"`` or
            ``"Answer:"``; ``None`` reads the last number in the output.
    """

    after: str | None = None

    def __call__(self, example: Example, output: str, /) -> Score:
        expected = _NUMBER.fullmatch(example.answer.strip())
        if expected is None:
            raise ValueError(f"NumericAnswer needs a numeric answer, got {example.answer!r}")
        found = self._read(output)
        if found is None:
            return Score(0.0, None)
        correct = _decimal(found) == _decimal(expected.group())
        return Score(1.0 if correct else 0.0, found)

    def _read(self, output: str) -> str | None:
        if self.after is None:
            numbers = _NUMBER.findall(output)
            return numbers[-1] if numbers else None
        _, marker, tail = output.rpartition(self.after)
        if not marker:
            return None
        first = _NUMBER.search(tail)
        return first.group() if first else None


def _decimal(number: str) -> decimal.Decimal:
    return decimal.Decimal(number.replace(",", ""))


@dataclasses.dataclass(frozen=True, slots=True)
class MultipleChoice:
    """Correct when the chosen letter is the answer's letter.

    The chosen letter comes from the last ``Answer:`` line (``Answer`` in any
    case) whose rest is a choice's letter, optionally in parentheses or followed
    by ``)`` or ``.``, and then either nothing or exactly that choice's text: so
    ``Answer: C``, ``Answer: (C).``, ``Answer: C) 4`` and ``Answer: C. 4`` all
    choose C when C is ``4``. Any other text after the letter, as in
    ``Answer: A careful reading...``, makes the line not an answer. Without
    such a line, an output that is a single letter (``B``, ``(B)``, ``B.``)
    is the choice. Only the letters of the example's choices count, and ``*``
    emphasis is ignored. `multiple_choice_prompt` asks for this format.
    """

    def __call__(self, example: Example, output: str, /) -> Score:
        if not example.choices:
            raise ValueError("MultipleChoice scores multiple-choice examples; this one has none")
        text = output.replace("*", "")
        lines = re.findall(r"(?i:answer)[^\S\n]*:([^\n]*)", text)
        answered = [_choice_on_line(rest, example.choices) for rest in lines]
        chosen = next((letter for letter in reversed(answered) if letter is not None), None)
        if chosen is None:
            chosen = _choice_on_line(text, example.choices, allow_option_text=False)
        return Score(1.0 if chosen == example.answer else 0.0, chosen)


_LETTER_THEN_TEXT = re.compile(r"\(?([A-Z])\)?\.?(?:\s+(.*))?", re.DOTALL)


def _choice_on_line(
    text: str, choices: tuple[str, ...], *, allow_option_text: bool = True
) -> str | None:
    """The letter ``text`` chooses: a letter, then nothing or that choice's own text."""
    match = _LETTER_THEN_TEXT.fullmatch(text.strip())
    if match is None:
        return None
    letter, option_text = match.groups()
    index = LETTERS.index(letter)
    if index >= len(choices):
        return None
    if option_text is not None and (
        not allow_option_text or option_text.strip() != choices[index].strip()
    ):
        return None
    return str(letter)


def multiple_choice_prompt(example: Example) -> str:
    """Formats a multiple-choice example in the form `MultipleChoice` reads.

    Raises:
        ValueError: The example has no choices.
    """
    if not example.choices:
        raise ValueError("this example has no choices")
    options = [
        f"{letter}) {choice}" for letter, choice in zip(LETTERS, example.choices, strict=False)
    ]
    instruction = (
        'End your response with a line of the form "Answer: X", '
        "where X is the letter of the correct choice."
    )
    return "\n".join([example.question, "", *options, "", instruction])
