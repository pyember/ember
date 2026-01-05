"""Extractors used by composed evaluators."""

from __future__ import annotations

import abc
import re
from collections.abc import Sequence
from typing import Generic, TypeVar

T_out = TypeVar("T_out")
T_truth = TypeVar("T_truth")


class IOutputExtractor(Generic[T_out, T_truth], metaclass=abc.ABCMeta):
    """Convert raw system output into a comparable form."""

    @abc.abstractmethod
    def extract(self, system_output: T_out, **kwargs: object) -> T_truth:
        raise NotImplementedError


class RegexExtractor(IOutputExtractor[str, str]):
    """Extract the first capturing group from a regex match."""

    def __init__(self, pattern: str) -> None:
        self.compiled_pattern: re.Pattern[str] = re.compile(pattern)

    def extract(self, system_output: str, **kwargs: object) -> str:
        match = self.compiled_pattern.search(system_output)
        if match is None:
            return ""
        return str(match.group(1))


class FinalLetterExtractor(IOutputExtractor[str, str]):
    """Extract the final multiple-choice letter from free-form text."""

    _MARKDOWN_PATTERN: re.Pattern[str] = re.compile(r"\*\*|\*|__|_|~~|`")

    def __init__(self, valid_letters: str = "ABCD") -> None:
        collected = [letter.upper() for letter in valid_letters if letter.isalpha()]
        if not collected:
            collected = list("ABCD")
        self.valid_letters = tuple(collected)
        letter_class = "".join(self.valid_letters)
        self._patterns: Sequence[re.Pattern[str]] = (
            re.compile(rf"\\boxed\{{([{letter_class}])\}}", re.IGNORECASE),
            re.compile(
                rf"final\s*answer\s*(?:is\s+|[:=]\s*)\(?([{letter_class}])\)?",
                re.IGNORECASE,
            ),
            re.compile(
                rf"(?:the\s+)?(?:correct\s+)?(?:answer|statement|option)\s+is\s+"
                rf"\(?([{letter_class}])\)?(?:\s|$|\.)",
                re.IGNORECASE,
            ),
            re.compile(
                rf"(?:answer|choice)\s*[:=]\s*\(?([{letter_class}])\)?"
                rf"(?!\s*(?:and|/|,)\s*[{letter_class}])",
                re.IGNORECASE,
            ),
            re.compile(
                rf"option\s+([{letter_class}])\s+is\s+(?:the\s+)?correct",
                re.IGNORECASE,
            ),
            re.compile(
                rf"(?:i\s+)?(?:would\s+)?choose\s+\(?([{letter_class}])\)?",
                re.IGNORECASE,
            ),
            re.compile(
                rf"\b([{letter_class}])\s+is\s+(?:the\s+)?(?:best|correct|right)\s+"
                rf"(?:answer|choice|option)",
                re.IGNORECASE,
            ),
            re.compile(rf"^([{letter_class}])[\.\)]\s+\S", re.IGNORECASE),
            re.compile(rf"^[ \t]*([{letter_class}])[ \t]*$", re.MULTILINE),
            re.compile(rf"\(([{letter_class}])\)\s*\.?\s*$"),
        )

    def _strip_markdown(self, text: str) -> str:
        return self._MARKDOWN_PATTERN.sub("", text)

    def extract(self, system_output: str, **kwargs: object) -> str:
        text = self._strip_markdown(system_output.strip())
        if not text:
            return ""
        for pattern in self._patterns:
            match = pattern.search(text)
            if match:
                letter = match.group(1).upper()
                if letter in self.valid_letters:
                    return letter
        return ""
