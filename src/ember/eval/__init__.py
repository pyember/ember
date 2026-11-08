"""Evaluation runs that record every request's outcome.

`run` applies a task (any callable, including an operator) to each item,
scores the output, and writes JSON-lines records of every item and every
request attempt, retries included, as it goes. `read_run` reads them back, and
the `Summary` of a run counts failures, retries, unpriced and unmeasured calls
next to the score instead of leaving them out.

Example:
    >>> from ember.data import Example
    >>> from ember.eval import NumericAnswer
    >>> NumericAnswer()(Example("2+2?", "4"), "Two plus two is 4.")
    Score(value=1.0, extracted='4')
"""

from ember.eval.encode import to_json
from ember.eval.jsonl import SCHEMA_VERSION, read_run
from ember.eval.records import (
    Abort,
    AttemptRecord,
    Completed,
    Failed,
    ItemRecord,
    RunEnd,
    RunHeader,
    RunRecords,
    Scored,
    Summary,
    summarize,
)
from ember.eval.run import run
from ember.eval.scorers import (
    ExactMatch,
    MultipleChoice,
    NumericAnswer,
    Score,
    Scorer,
    multiple_choice_prompt,
)

__all__ = [
    "SCHEMA_VERSION",
    "Abort",
    "AttemptRecord",
    "Completed",
    "ExactMatch",
    "Failed",
    "ItemRecord",
    "MultipleChoice",
    "NumericAnswer",
    "RunEnd",
    "RunHeader",
    "RunRecords",
    "Score",
    "Scored",
    "Scorer",
    "Summary",
    "multiple_choice_prompt",
    "read_run",
    "run",
    "summarize",
    "to_json",
]
