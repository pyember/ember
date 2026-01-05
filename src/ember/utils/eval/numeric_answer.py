"""Evaluators for integer-answer datasets (for example AIME)."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeAlias

from ember._internal.exceptions import DataError
from ember.utils.eval.base_evaluator import EvaluationResult, IEvaluator

ExtractionMetadata: TypeAlias = dict[str, object]
ExtractionResult: TypeAlias = tuple[list[int], ExtractionMetadata]

_AIME_DIGITS_PATTERN = r"(?<!\d)(\d{1,3})(?!\d)"


def _parse_aime_answer(answer_str: str) -> int:
    normalized = answer_str.strip()
    try:
        value = int(normalized)
    except ValueError as exc:
        raise DataError(
            message="AIME reference answer must be an integer string",
            context={"correct_answer": answer_str},
            recovery_hint="Provide correct_answer as a base-10 integer between 0 and 999.",
        ) from exc

    if not 0 <= value <= 999:
        raise DataError(
            message="AIME reference answer must be between 0 and 999",
            context={"correct_answer": answer_str, "parsed_value": value},
            recovery_hint="Provide correct_answer as a base-10 integer between 0 and 999.",
        )

    return value


class IAnswerExtractor(ABC):
    """Extract candidate integer answers from text."""

    @abstractmethod
    def extract(self, text: str) -> ExtractionResult:
        raise NotImplementedError


class RegexAnswerExtractor(IAnswerExtractor):
    """Extract answers using a regular expression."""

    def __init__(self, pattern: str, flags: int = re.IGNORECASE) -> None:
        self.pattern: re.Pattern[str] = re.compile(pattern, flags)
        self.name = self.__class__.__name__

    def extract(self, text: str) -> ExtractionResult:
        matches = self.pattern.findall(text)
        valid_numbers: list[int] = []

        for match in matches:
            groups: Sequence[str]
            if isinstance(match, tuple):
                groups = match
            else:
                groups = (match,)

            for group in groups:
                candidate = group.strip()
                if not candidate:
                    continue
                try:
                    valid_numbers.append(int(candidate))
                except ValueError:
                    continue

        return valid_numbers, {"method": self.name, "matches": matches}


class FinalAnswerExtractor(RegexAnswerExtractor):
    def __init__(self) -> None:
        pattern = rf"(?:final\s+answer\s*(?:is|:|=))\s*{_AIME_DIGITS_PATTERN}"
        super().__init__(pattern)


class TheAnswerExtractor(RegexAnswerExtractor):
    def __init__(self) -> None:
        pattern = rf"(?:the\s+answer\s*(?:is|:|=))\s*{_AIME_DIGITS_PATTERN}"
        super().__init__(pattern)


class EqualsExtractor(RegexAnswerExtractor):
    def __init__(self) -> None:
        pattern = rf"=\s*{_AIME_DIGITS_PATTERN}"
        super().__init__(pattern)


class ThereforeExtractor(RegexAnswerExtractor):
    def __init__(self) -> None:
        pattern = rf"(?:therefore,?\s+(?:the\s+)?answer\s*(?:is|:|=))\s*{_AIME_DIGITS_PATTERN}"
        super().__init__(pattern)


class GetAnswerExtractor(RegexAnswerExtractor):
    def __init__(self) -> None:
        pattern = (
            rf"(?:we|I)\s+get\s+{_AIME_DIGITS_PATTERN}\s+as\s+"
            r"(?:our|the)\s+(?:final\s+)?answer"
        )
        super().__init__(pattern)


class HashAnswerExtractor(RegexAnswerExtractor):
    def __init__(self) -> None:
        pattern = r"#{3}\s*" + _AIME_DIGITS_PATTERN
        super().__init__(pattern)


class GenericNumberExtractor(RegexAnswerExtractor):
    def __init__(self) -> None:
        super().__init__(_AIME_DIGITS_PATTERN)

    def extract(self, text: str) -> ExtractionResult:
        numbers, metadata = super().extract(text)

        valid_numbers = [num for num in numbers if 0 <= num <= 999]

        metadata["original_count"] = len(numbers)
        metadata["valid_count"] = len(valid_numbers)

        return valid_numbers, metadata


class NumericAnswerEvaluator(IEvaluator[str, str]):
    """Evaluator for integer answers embedded in text."""

    def __init__(self, extract_pattern: str | None = None) -> None:
        if extract_pattern is not None:
            self.extractor = RegexAnswerExtractor(extract_pattern)
        else:
            default_pattern = r"(?:answer|result)?\s*(?:is|=|:)?\s*(-?\d+)"
            self.extractor = RegexAnswerExtractor(default_pattern)

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: object
    ) -> EvaluationResult:
        try:
            expected = int(correct_answer.strip())
        except ValueError as exc:
            raise DataError(
                message="NumericAnswerEvaluator reference answer must be an integer string",
                context={"correct_answer": correct_answer},
                recovery_hint="Provide correct_answer as a base-10 integer string.",
            ) from exc

        extracted_numbers, metadata = self.extractor.extract(system_output)

        is_correct = expected in extracted_numbers

        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            metadata={**metadata, "expected": expected, "found": is_correct},
        )


class AIMEAnswerEvaluator(IEvaluator[str, str]):
    """Evaluator for AIME-style answers (0-999)."""

    def __init__(self, custom_extractors: Sequence[IAnswerExtractor] | None = None) -> None:
        self.primary_extractors = list(custom_extractors) if custom_extractors else [
            HashAnswerExtractor(),
            FinalAnswerExtractor(),
            TheAnswerExtractor(),
            ThereforeExtractor(),
            GetAnswerExtractor(),
            EqualsExtractor(),
        ]

        self.fallback_extractor = GenericNumberExtractor()

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: object
    ) -> EvaluationResult:
        expected = _parse_aime_answer(correct_answer)

        for extractor in self.primary_extractors:
            numbers, metadata = extractor.extract(system_output)

            if numbers:
                is_correct = expected in numbers
                return EvaluationResult(
                    is_correct=is_correct,
                    score=1.0 if is_correct else 0.0,
                    metadata={
                        "extracted_method": "final_pattern",
                        "extractor": metadata["method"],
                        "extracted_values": numbers,
                        "expected": expected,
                    },
                )

        numbers, metadata = self.fallback_extractor.extract(system_output)

        is_correct = expected in numbers
        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            metadata={
                "extracted_method": "fallback_pattern",
                "extractor": metadata["method"],
                "extracted_values": numbers,
                "expected": expected,
            },
        )
