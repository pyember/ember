"""The datasets ember knows by name.

Each entry pins the Hub repository, configuration, default split and commit,
names the Parquet file that holds each split, and maps rows to examples with a
function written for that dataset's schema. A row that does not match the
schema raises instead of being skipped or guessed at.
"""

from collections.abc import Mapping
from types import MappingProxyType

from ember.data.example import LETTERS, Dataset, Example, Row

__all__ = ["DATASETS"]


def _text(row: Row, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"field {key!r} must be a string, got {type(value).__name__}")
    return value


def _texts(row: Row, key: str) -> tuple[str, ...]:
    value = row.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"field {key!r} must be a list of strings")
    return tuple(value)


def _mmlu_example(row: Row) -> Example:
    """MMLU: ``answer`` is the index of the correct entry in ``choices``."""
    choices = _texts(row, "choices")
    answer = row.get("answer")
    if not isinstance(answer, int) or isinstance(answer, bool) or not 0 <= answer < len(choices):
        raise ValueError(
            f"field 'answer' must index into the {len(choices)} choices, got {answer!r}"
        )
    return Example(
        question=_text(row, "question"),
        answer=LETTERS[answer],
        choices=choices,
        metadata=MappingProxyType({"subject": _text(row, "subject")}),
    )


def _gsm8k_example(row: Row) -> Example:
    """GSM8K: ``answer`` is a worked solution whose last line is ``#### <number>``."""
    solution, marker, final = _text(row, "answer").rpartition("####")
    if not marker:
        raise ValueError("field 'answer' has no '####' line giving the final answer")
    return Example(
        question=_text(row, "question"),
        answer=final.strip(),
        metadata=MappingProxyType({"solution": solution.strip()}),
    )


# Each revision is the repository's commit as read from
# https://huggingface.co/api/datasets/<id> ("sha") on 2026-11-07. Moving a pin
# changes what a named dataset means, so it is a deliberate, reviewed change.
# At those commits every split of every configuration (MMLU's 59, GSM8K's 2) is
# one file at this path, as listed by
# https://huggingface.co/api/datasets/<id>/tree/<sha>/<config> on 2026-11-09.
_ONE_FILE_PER_SPLIT = "{config}/{split}-00000-of-00001.parquet"

DATASETS: Mapping[str, Dataset] = MappingProxyType(
    {
        "mmlu": Dataset(
            "cais/mmlu",
            "all",
            "test",
            _mmlu_example,
            revision="c30699e8356da336a370243923dbaf21066bb9fe",
            parquet_file=_ONE_FILE_PER_SPLIT,
        ),
        "gsm8k": Dataset(
            "openai/gsm8k",
            "main",
            "test",
            _gsm8k_example,
            revision="740312add88f781978c0658806c59bc2815b9866",
            parquet_file=_ONE_FILE_PER_SPLIT,
        ),
    }
)
"""Datasets `ember.data.stream` accepts by name, each pinned to a Hub commit."""
