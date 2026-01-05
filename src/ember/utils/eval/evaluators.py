"""Evaluator implementations used by Ember."""

from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Generic, TypeVar

from .base_evaluator import EvaluationResult, IEvaluator
from .extractors import FinalLetterExtractor, IOutputExtractor, RegexExtractor

T_out = TypeVar("T_out")
T_extracted = TypeVar("T_extracted")
T_truth = TypeVar("T_truth")


class ComposedEvaluator(
    IEvaluator[T_out, T_truth],
    Generic[T_out, T_extracted, T_truth],
):
    """Apply an extractor before delegating to the base evaluator."""

    def __init__(
        self,
        extractor: IOutputExtractor[T_out, T_extracted],
        base_evaluator: IEvaluator[T_extracted, T_truth],
    ) -> None:
        self.extractor = extractor
        self.base_evaluator = base_evaluator

    def evaluate(
        self, system_output: T_out, correct_answer: T_truth, **kwargs: object
    ) -> EvaluationResult:
        extracted_value = self.extractor.extract(system_output, **kwargs)
        return self.base_evaluator.evaluate(extracted_value, correct_answer, **kwargs)

class ExactMatchEvaluator(IEvaluator[str, str]):
    """Check if two strings match after stripping whitespace."""

    def __init__(
        self,
        compare_fn: Callable[[str, str], bool] | None = None,
        *,
        case_sensitive: bool = True,
    ) -> None:
        self.case_sensitive = case_sensitive
        self.compare_fn = compare_fn or self._default_compare

    def _default_compare(self, str1: str, str2: str) -> bool:
        """Compare two strings with configurable case sensitivity."""

        left = str1.strip()
        right = str2.strip()
        if not self.case_sensitive:
            left = left.lower()
            right = right.lower()
        return left == right

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: object
    ) -> EvaluationResult:
        is_correct = self.compare_fn(system_output, correct_answer)
        score = 1.0 if is_correct else 0.0
        return EvaluationResult(is_correct=is_correct, score=score)


class PythonLiteralEqualityEvaluator(IEvaluator[str, str]):
    """Compare values by parsing Python literals when possible."""

    def __init__(self, float_tol: float | None = None) -> None:
        self.float_tol = float_tol

    def _try_eval(self, value: str) -> tuple[bool, object]:
        text = value.strip()
        try:
            return True, ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return False, text

    def _eq(self, left: object, right: object) -> bool:
        if isinstance(left, float) and isinstance(right, float) and self.float_tol is not None:
            return math.isclose(left, right, rel_tol=0.0, abs_tol=self.float_tol)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                return False
            return all(
                self._eq(l_item, r_item)
                for l_item, r_item in zip(left, right, strict=True)
            )
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                return False
            return all(self._eq(left[key], right[key]) for key in left)
        return left == right

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: object
    ) -> EvaluationResult:
        pred_parsed, pred_value = self._try_eval(system_output)
        gold_parsed, gold_value = self._try_eval(correct_answer)

        if pred_parsed and gold_parsed:
            is_correct = self._eq(pred_value, gold_value)
        else:
            pred_text = str(pred_value).strip()
            gold_text = str(gold_value).strip()
            is_correct = pred_text == gold_text

        return EvaluationResult(is_correct=is_correct, score=1.0 if is_correct else 0.0)


class NumericToleranceEvaluator(IEvaluator[float, float]):
    """Check whether a numeric output is within a tolerance of the expected value."""

    def __init__(self, tolerance: float = 0.01) -> None:
        self.tolerance = tolerance

    def evaluate(
        self, system_output: float, correct_answer: float, **kwargs: object
    ) -> EvaluationResult:
        difference = abs(system_output - correct_answer)
        rounded_diff = round(difference, 8)
        is_correct = rounded_diff <= self.tolerance
        base = abs(correct_answer) if correct_answer != 0 else 1.0
        score = max(0.0, 1.0 - rounded_diff / base)
        return EvaluationResult(is_correct=is_correct, score=score, metadata={"diff": rounded_diff})


class CodeExecutionEvaluator(IEvaluator[str, str]):
    """Execute Python code and compare stdout (unsafe; opt-in required)."""

    def __init__(self, timeout: float = 5.0, *, allow_unsafe: bool = False) -> None:
        self.timeout = timeout
        self.allow_unsafe = allow_unsafe

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: object
    ) -> EvaluationResult:
        if not self.allow_unsafe:
            raise RuntimeError(
                "CodeExecutionEvaluator is disabled by default because it executes arbitrary code. "
                "Pass allow_unsafe=True (trusted inputs only) or use a sandboxed evaluator."
            )

        process_result: subprocess.CompletedProcess[str]
        try:
            process_result = subprocess.run(
                args=[sys.executable, "-c", system_output],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as timeout_error:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                metadata={"error": f"TimeoutExpired: {str(timeout_error)}"},
            )

        stdout_str = process_result.stdout.strip()
        expected_str = correct_answer.strip()
        is_correct = stdout_str == expected_str
        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            metadata={
                "stdout": process_result.stdout,
                "stderr": process_result.stderr,
                "exit_code": process_result.returncode,
            },
        )


class MultipleChoiceStrictEvaluator(IEvaluator[str, str]):
    """Robust multiple-choice evaluator that focuses on final answer letters."""

    def __init__(self, valid_letters: str = "ABCDEFGHIJ") -> None:
        self.valid_letters = tuple(ch for ch in valid_letters if ch.isalpha())
        self.extractor = FinalLetterExtractor(valid_letters="".join(self.valid_letters))

    def _map_choice_text(self, text: str, choices: Mapping[str, str] | None) -> str:
        if not choices:
            return ""
        normalized_map: dict[str, str] = {}
        for label, option_text in choices.items():
            normalized_map[option_text.strip().lower()] = label.strip().upper()
        lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
        for candidate in reversed(lines):
            mapped = normalized_map.get(candidate)
            if mapped:
                return mapped
        candidate = text.strip().lower()
        return normalized_map.get(candidate, "")

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: object
    ) -> EvaluationResult:
        expected = correct_answer.strip().upper()
        choices = kwargs.get("choices")
        extracted = self.extractor.extract(system_output)
        metadata: dict[str, object] = {"extracted": extracted}

        if not extracted and isinstance(choices, Mapping):
            mapped = self._map_choice_text(system_output, choices)
            if mapped:
                extracted = mapped
                metadata["mapped_from_text"] = True

        if not extracted:
            return EvaluationResult(is_correct=False, score=0.0, metadata=metadata)

        extracted = extracted.strip().upper()
        if extracted not in self.valid_letters:
            metadata["invalid_letter"] = extracted
            return EvaluationResult(is_correct=False, score=0.0, metadata=metadata)

        is_correct = extracted == expected
        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            metadata=metadata,
        )


class YesNoEvaluator(IEvaluator[str, str]):
    """Evaluator for binary yes/no style tasks with common synonym handling."""

    _ALIASES = {
        "yes": "yes",
        "y": "yes",
        "true": "yes",
        "correct": "yes",
        "pass": "yes",
        "no": "no",
        "n": "no",
        "false": "no",
        "fail": "no",
    }

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: object
    ) -> EvaluationResult:
        def normalize(value: str) -> str:
            key = value.strip().lower()
            return self._ALIASES.get(key, key)

        pred = normalize(system_output)
        gold = normalize(correct_answer)
        is_correct = pred == gold and pred in {"yes", "no"}
        return EvaluationResult(is_correct=is_correct, score=1.0 if is_correct else 0.0)


def _strip_latex_bbeh(response: str) -> str:
    text = response.strip()
    if text.startswith("$") and text.endswith("$"):
        text = text[1:-1]
    if "boxed{" in text and text.endswith("}"):
        text = text[:-1].split("boxed{")[-1]
    if "text{" in text and text.endswith("}"):
        text = text[:-1].split("text{")[-1]
    if "texttt{" in text and text.endswith("}"):
        text = text[:-1].split("texttt{")[-1]
    return text


def _extract_answer_bbeh(sample: str) -> str:
    answer_prefixes = [
        "The final answer is:",
        "The final answer is ",
        "The answer is:",
        "The answer is ",
        "Final answer:",
        "Final answer is:",
        "ANSWER:",
        "Answer:",
    ]

    answer = sample
    prefix_found = False

    for answer_prefix in answer_prefixes:
        if answer_prefix in answer:
            answer = answer.split(answer_prefix)[-1].strip()
            prefix_found = True
            break

    if not prefix_found:
        start_answer_match = re.match(
            r"^\s*\**\s*(yes|no|true|false)\s*\**\s*[.,]",
            sample,
            re.IGNORECASE,
        )
        if start_answer_match:
            answer = start_answer_match.group(1).lower()
            prefix_found = True

    if not prefix_found:
        mc_patterns = [
            r"\*\*\(([A-Za-z])\)\*\*",
            r"(?:only|option|expression|answer|choice|statement)\s+\**\(([A-Za-z])\)\**",
            r"is\s+\**\(([A-Za-z])\)\**",
            r"^\s*\(([A-Za-z])\)\s*$",
        ]
        for pattern in mc_patterns:
            match = re.search(pattern, sample, re.IGNORECASE)
            if match:
                answer = f"({match.group(1).upper()})"
                prefix_found = True
                break

    answer = _strip_latex_bbeh(answer)
    answer = answer.lower().strip()
    answer = answer.replace(", ", ",").replace("**", "")
    answer = answer.split("\n")[0]
    if answer.endswith("."):
        answer = answer[:-1]

    return answer


def _fuzzy_match_bbeh(prediction: str, reference: str) -> bool:
    if prediction == reference:
        return True

    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        if prediction[1] == reference:
            return True
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        if reference[1] == prediction:
            return True

    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass

    if prediction.replace("'", "") == reference.replace("'", ""):
        return True

    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True

    if prediction.endswith("?") and prediction[:-1] == reference:
        return True
    if reference.endswith("?") and reference[:-1] == prediction:
        return True

    return False


class BBEHEvaluator(IEvaluator[str, str]):
    """Evaluator for BBEH using official answer extraction and fuzzy matching."""

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: object
    ) -> EvaluationResult:
        gold = correct_answer.strip()
        if gold.startswith("[") and gold.endswith("]"):
            raise ValueError(
                f"BBEH ground truth must be scalar, got list-like: {gold!r}"
            )

        extracted = _extract_answer_bbeh(system_output)
        reference = gold.lower().replace(", ", ",")
        is_correct = _fuzzy_match_bbeh(extracted, reference)
        metadata: dict[str, str] = {"extracted": extracted, "reference": reference}
        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            metadata=metadata,
        )


class TokenListEvaluator(IEvaluator[str, Sequence[str]]):
    """Compare ordered token sequences extracted from text outputs."""

    def __init__(self, separator_pattern: str = r"[,\s]+") -> None:
        self._separator = re.compile(separator_pattern)

    def _to_tokens(self, value: object) -> list[str]:
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value]
        text = str(value).strip()
        if not text:
            return []
        return [token for token in self._separator.split(text) if token]

    def _final_nonempty_line(self, text: str) -> str:
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if stripped:
                return stripped
        return text.strip()

    def evaluate(
        self, system_output: str, correct_answer: Sequence[str] | str, **kwargs: object
    ) -> EvaluationResult:
        gold_tokens = self._to_tokens(correct_answer)
        prediction_line = self._final_nonempty_line(system_output)
        predicted_tokens = self._to_tokens(prediction_line)
        is_correct = predicted_tokens == gold_tokens
        metadata = {"pred_tokens": predicted_tokens, "gold_tokens": gold_tokens}
        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            metadata=metadata,
        )


class JsonValueEvaluator(IEvaluator[str, str]):
    """Compare JSON-encoded results focusing on a single key (default ``result``)."""

    def __init__(self, key: str = "result", float_tol: float | None = None) -> None:
        self.key = key
        self.float_tol = float_tol

    def _parse(self, value: object) -> tuple[bool, object]:
        if isinstance(value, Mapping):
            return True, value
        if isinstance(value, str):
            try:
                return True, json.loads(value)
            except json.JSONDecodeError:
                return False, value.strip()
        return False, value

    def _compare_scalar(self, left: object, right: object) -> bool:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if self.float_tol is not None:
                return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=self.float_tol)
            return float(left) == float(right)
        return left == right

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: object
    ) -> EvaluationResult:
        pred_ok, pred_value = self._parse(system_output)
        gold_ok, gold_value = self._parse(correct_answer)

        if (
            pred_ok
            and isinstance(pred_value, Mapping)
            and self.key in pred_value
            and gold_ok
            and isinstance(gold_value, Mapping)
            and self.key in gold_value
        ):
            pred_result = pred_value[self.key]
            gold_result = gold_value[self.key]
            is_correct = self._compare_scalar(pred_result, gold_result)
        else:
            is_correct = system_output.strip() == str(correct_answer).strip()

        return EvaluationResult(is_correct=is_correct, score=1.0 if is_correct else 0.0)


class PartialRegexEvaluator(ComposedEvaluator[str, str, str]):
    """Extract with a regex, then check exact match."""

    def __init__(self, pattern: str) -> None:
        extractor = RegexExtractor(pattern)
        evaluator = ExactMatchEvaluator()
        super().__init__(extractor=extractor, base_evaluator=evaluator)
