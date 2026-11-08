"""What an evaluation run records, and the summary computed from it.

A run produces one `ItemRecord` per item and one `AttemptRecord` per request
attempt made while that item ran, retries included. The `Summary` is computed
from the item records alone, so a run read back from its file summarizes to
exactly what the run reported.
"""

import collections
import dataclasses
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from ember.data import Source
from ember.errors import ErrorKind, ModelError
from ember.eval.scorers import Score
from ember.llm import FinishReason, ModelRef, Money, Result, Usage
from ember.llm.observe import Attempt
from ember.llm.request import JsonValue

__all__ = [
    "Abort",
    "AttemptRecord",
    "Completed",
    "CutOff",
    "Failed",
    "ItemRecord",
    "RunEnd",
    "RunHeader",
    "RunRecords",
    "Scored",
    "Summary",
    "summarize",
]


@dataclasses.dataclass(frozen=True, slots=True)
class Failed:
    """An exception that ended an attempt, an item or a run.

    Attributes:
        error_type: The exception's class name, for example ``"ModelError"``.
        kind: The `ErrorKind` of a `ModelError`; ``None`` for other exceptions.
        status: The HTTP status of a `ModelError`, when it had one.
        message: The exception's message. `ModelError` messages never contain
            upstream bodies or credentials.
    """

    error_type: str
    kind: ErrorKind | None
    status: int | None
    message: str

    @classmethod
    def of(cls, error: BaseException) -> "Failed":
        """Records ``error``."""
        if isinstance(error, ModelError):
            return cls(type(error).__name__, error.kind, error.status, str(error))
        return cls(type(error).__name__, None, None, str(error))

    @property
    def label(self) -> str:
        """``"ModelError:rate_limited"`` for model errors, else the class name."""
        return self.error_type if self.kind is None else f"{self.error_type}:{self.kind.value}"


@dataclasses.dataclass(frozen=True, slots=True)
class Completed:
    """An attempt that returned a result.

    Attributes:
        finish_reason: Why generation ended.
        usage: Tokens the provider reported.
        cost: The call's cost, or ``None`` when the model has no known price.
    """

    finish_reason: FinishReason
    usage: Usage
    cost: Money | None


@dataclasses.dataclass(frozen=True, slots=True)
class AttemptRecord:
    """One attempt to send one request.

    Attributes:
        call_id: Identifies the model call; retries of a call share it.
        number: 1 for the first attempt, 2 for the first retry, and so on.
        model: The model the attempt was sent to.
        latency_s: Seconds from sending to the last byte or the failure.
        request_id: The provider's request id, when it sent one.
        will_retry: Whether the client sent another attempt after this one.
        outcome: The result's accounting, or the error.
    """

    call_id: str
    number: int
    model: ModelRef
    latency_s: float
    request_id: str | None
    will_retry: bool
    outcome: Completed | Failed

    @classmethod
    def of(cls, attempt: Attempt) -> "AttemptRecord":
        """Records an attempt reported by `ember.llm.observe`."""
        result = attempt.outcome
        if isinstance(result, Result):
            outcome: Completed | Failed = Completed(result.finish_reason, result.usage, result.cost)
            request_id = result.request_id
        else:
            outcome, request_id = Failed.of(result), result.request_id
        return cls(
            attempt.call_id,
            attempt.number,
            attempt.model,
            attempt.latency_s,
            request_id,
            attempt.will_retry,
            outcome,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Scored:
    """An item whose task returned an output that was scored.

    Attributes:
        output: The task's output, as recorded.
        score: The scorer's verdict.
    """

    output: JsonValue
    score: Score


@dataclasses.dataclass(frozen=True, slots=True)
class ItemRecord:
    """One item of a run.

    Attributes:
        index: The item's position in the run's input, from 0.
        item: The item, as recorded.
        latency_s: Seconds the task and scorer took for this item.
        outcome: The scored output, or the error the task raised.
        attempts: Every request attempt made for this item, retries included.
    """

    index: int
    item: JsonValue
    latency_s: float
    outcome: Scored | Failed
    attempts: tuple[AttemptRecord, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class RunHeader:
    """The first record of a run.

    Attributes:
        run_id: A random id for the run.
        started_at: Start time, ISO 8601 in UTC.
        ember_version: The ember version that ran it.
        scorer: The scorer's repr, which for built-in scorers names their rule.
        source: Where the items came from, when they were an
            `ember.data.Stream`; ``None`` otherwise.
    """

    run_id: str
    started_at: str
    ember_version: str
    scorer: str
    source: Source | None


@dataclasses.dataclass(frozen=True, slots=True)
class Abort:
    """Why a run stopped before its last item.

    Attributes:
        index: The index of the item that was running or being read.
        error: The exception that stopped the run.
    """

    index: int
    error: Failed


@dataclasses.dataclass(frozen=True, slots=True)
class RunEnd:
    """The last record of a run that finished writing.

    Attributes:
        elapsed_s: Seconds from the header to this record.
        abort: Why the run stopped early, or ``None`` when every item ran.
    """

    elapsed_s: float
    abort: Abort | None


@dataclasses.dataclass(frozen=True, slots=True)
class CutOff:
    """A run file that stops before its end record, because the run died.

    Attributes:
        partial_record_dropped: The file ended inside a record, and that
            incomplete last line was left out.
    """

    partial_record_dropped: bool


@dataclasses.dataclass(frozen=True, slots=True)
class Summary:
    """Counts over a run's items and attempts. Failures are counted, never dropped.

    Attributes:
        items: Items recorded.
        scored: Items whose output was scored.
        failed: Items whose task raised; they have no score.
        mean_score: Mean score over the scored items only, or ``None`` when
            none was scored. Read it together with ``failed``.
        failures: Failed items by `Failed.label`.
        calls: Distinct model calls.
        attempts: Request attempts, retries included.
        retries: Attempts after the first of their call.
        failed_attempts: Attempts that ended in an error.
        failed_calls: Calls whose last attempt ended in an error.
        unpriced_calls: Completed calls with no known price; ``cost`` leaves
            them out.
        unmeasured_calls: Completed calls whose provider reported no token
            counts; ``usage`` leaves them out.
        cost: Total cost of the priced completed calls.
        usage: Total tokens of the measured completed calls.
    """

    items: int
    scored: int
    failed: int
    mean_score: float | None
    failures: Mapping[str, int]
    calls: int
    attempts: int
    retries: int
    failed_attempts: int
    failed_calls: int
    unpriced_calls: int
    unmeasured_calls: int
    cost: Money
    usage: Usage

    def __str__(self) -> str:
        mean = "no item scored" if self.mean_score is None else f"mean {self.mean_score:.4f}"
        failures = ", ".join(f"{label} {count}" for label, count in self.failures.items())
        lines = [
            f"items {self.items}: scored {self.scored} ({mean}), failed {self.failed}"
            + (f" ({failures})" if failures else ""),
            f"calls {self.calls}: attempts {self.attempts}, retries {self.retries}, "
            f"failed attempts {self.failed_attempts}, failed calls {self.failed_calls}",
            f"cost {self.cost} (unpriced calls {self.unpriced_calls}), "
            f"tokens in {self.usage.prompt_tokens} out {self.usage.output} "
            f"(unmeasured calls {self.unmeasured_calls})",
        ]
        return "\n".join(lines)


@dataclasses.dataclass(frozen=True, slots=True)
class RunRecords:
    """Everything a run recorded.

    Attributes:
        header: The run's first record.
        items: The item records, in the order they finished.
        summary: Computed from ``items``.
        end: The last record, or `CutOff` when the file stops before it.
    """

    header: RunHeader
    items: tuple[ItemRecord, ...]
    summary: Summary
    end: RunEnd | CutOff


def summarize(items: Sequence[ItemRecord]) -> Summary:
    """Computes the summary of ``items``."""
    scores = [item.outcome.score.value for item in items if isinstance(item.outcome, Scored)]
    failures = collections.Counter(
        item.outcome.label for item in items if isinstance(item.outcome, Failed)
    )
    attempts = [attempt for item in items for attempt in item.attempts]
    completed = [a.outcome for a in attempts if isinstance(a.outcome, Completed)]
    last_attempts = _last_attempt_per_call(attempts)
    return Summary(
        items=len(items),
        scored=len(scores),
        failed=failures.total(),
        mean_score=math.fsum(scores) / len(scores) if scores else None,
        failures=MappingProxyType(dict(sorted(failures.items()))),
        calls=len(last_attempts),
        attempts=len(attempts),
        retries=sum(1 for attempt in attempts if attempt.number > 1),
        failed_attempts=len(attempts) - len(completed),
        failed_calls=sum(1 for a in last_attempts if isinstance(a.outcome, Failed)),
        unpriced_calls=sum(1 for outcome in completed if outcome.cost is None),
        unmeasured_calls=sum(1 for outcome in completed if not outcome.usage.measured),
        cost=sum((o.cost for o in completed if o.cost is not None), Money(0)),
        usage=sum((o.usage for o in completed if o.usage.measured), Usage()),
    )


def _last_attempt_per_call(attempts: Sequence[AttemptRecord]) -> list[AttemptRecord]:
    last: dict[str, AttemptRecord] = {}
    for attempt in attempts:
        seen = last.get(attempt.call_id)
        if seen is None or attempt.number > seen.number:
            last[attempt.call_id] = attempt
    return list(last.values())
