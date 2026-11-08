"""Running a task over items and recording everything that happened."""

import dataclasses
import datetime
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Generic, TypeVar

import ember
from ember.data import Stream
from ember.errors import ModelError
from ember.eval.encode import json_value, to_json
from ember.eval.jsonl import RunWriter
from ember.eval.records import (
    Abort,
    AttemptRecord,
    Failed,
    ItemRecord,
    RunEnd,
    RunHeader,
    RunRecords,
    Scored,
    summarize,
)
from ember.eval.scorers import Scorer
from ember.llm.observe import Attempt, observing
from ember.llm.request import JsonValue

__all__ = ["run"]

ItemT = TypeVar("ItemT")
OutputT = TypeVar("OutputT")


def run(
    task: Callable[[ItemT], OutputT],
    items: Iterable[ItemT],
    *,
    scorer: Scorer[ItemT, OutputT],
    records: Path,
    encode_item: Callable[[ItemT], JsonValue] = to_json,
    encode_output: Callable[[OutputT], JsonValue] = to_json,
    item_errors: tuple[type[Exception], ...] = (ModelError,),
) -> RunRecords:
    """Runs ``task`` on each item, scores each output, and records it all.

    For every item the run records the item, the output and its score, or the
    error the task raised; and for every request attempt made while the item
    ran (observed through `ember.llm.observe`), its call id, attempt number,
    model, latency, request id, outcome, cost and whether it was retried.
    Records go to ``records`` as JSON lines (see `ember.eval.jsonl`) as each
    item finishes, so a run that dies still leaves everything before it.

    Items run one after another, in order. When ``items`` is an
    `ember.data.Stream`, the run header records its `ember.data.Source`.

    Args:
        task: Produces an output for one item, for example by calling a model
            or an operator.
        items: The inputs, such as ``ember.data.stream("gsm8k")``.
        scorer: Scores one output against its item.
        records: A new file for the records; the run refuses to overwrite one.
        encode_item: Turns an item into JSON for the records; the default
            accepts `ember.data.Example` and JSON values.
        encode_output: Turns an output into JSON for the records; the default
            accepts JSON values.
        item_errors: Exception types that fail only the item that raised
            them, whether from the task, the scorer or ``encode_output``: the
            item is recorded as failed and the run goes on. Any other exception
            stops the run, after an end record naming it is written.

    Returns:
        The records, exactly as `ember.eval.read_run` reads them back.

    Raises:
        FileExistsError: ``records`` already exists.
        FileNotFoundError: The directory for ``records`` does not exist.
        Exception: Whatever the task, scorer, encoders or ``items`` raised,
            other than ``item_errors`` from the task.
    """
    runner = _ItemRunner(task, scorer, encode_item, encode_output, item_errors)
    header = RunHeader(
        run_id=uuid.uuid4().hex,
        started_at=datetime.datetime.now(datetime.UTC).isoformat(),
        ember_version=ember.__version__,
        scorer=repr(scorer),
        source=items.source if isinstance(items, Stream) else None,
    )
    started = time.perf_counter()
    done: list[ItemRecord] = []
    with RunWriter(records) as writer:
        writer.header(header)
        try:
            for index, item in enumerate(items):
                record = runner(index, item)
                writer.item(record)
                done.append(record)
        except BaseException as exc:
            abort = Abort(index=len(done), error=Failed.of(exc))
            writer.end(RunEnd(time.perf_counter() - started, abort), summarize(done))
            raise
        end = RunEnd(time.perf_counter() - started, None)
        summary = summarize(done)
        writer.end(end, summary)
    return RunRecords(header, tuple(done), summary, end)


@dataclasses.dataclass(frozen=True, slots=True)
class _ItemRunner(Generic[ItemT, OutputT]):
    """Runs and records one item; safe to call for several items concurrently."""

    task: Callable[[ItemT], OutputT]
    scorer: Scorer[ItemT, OutputT]
    encode_item: Callable[[ItemT], JsonValue]
    encode_output: Callable[[OutputT], JsonValue]
    item_errors: tuple[type[Exception], ...]

    def __call__(self, index: int, item: ItemT) -> ItemRecord:
        recorded_item = json_value(self.encode_item(item))
        attempts = _AttemptLog()
        started = time.perf_counter()
        with observing(attempts):
            try:
                outcome: Scored | Failed = self._score(item)
            except self.item_errors as exc:
                outcome = Failed.of(exc)
        latency_s = time.perf_counter() - started
        return ItemRecord(index, recorded_item, latency_s, outcome, attempts.records())

    def _score(self, item: ItemT) -> Scored:
        # Scoring and encoding read the output, and a deferred call raises its
        # error on first read, so both run inside the item's error handling.
        output = self.task(item)
        score = self.scorer(item, output)
        return Scored(json_value(self.encode_output(output)), score)


class _AttemptLog:
    """Observer for one item's attempts, which may be reported from worker threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[AttemptRecord] = []

    def on_attempt(self, attempt: Attempt) -> None:
        record = AttemptRecord.of(attempt)
        with self._lock:
            self._records.append(record)

    def records(self) -> tuple[AttemptRecord, ...]:
        with self._lock:
            return tuple(self._records)
