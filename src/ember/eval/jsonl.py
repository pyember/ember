"""The run file: one JSON object per line, schema version 1.

Every line carries ``"schema": 1`` and a ``"type"``:

- ``run``: the header (`RunHeader`), always first.
- ``item``: one `ItemRecord` with its attempts, written as the item finishes.
- ``end``: the `RunEnd` with the summary, always last. A file without it was
  cut off mid-run; everything before the cut is still readable, and an
  incomplete last line is dropped.

Item records appear in index order.

This module is the only code that writes or parses that format.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import IO, TypeVar

from ember.data import Source
from ember.errors import ErrorKind
from ember.eval.records import (
    Abort,
    AttemptRecord,
    Completed,
    CutOff,
    Failed,
    ItemRecord,
    RunEnd,
    RunHeader,
    RunRecords,
    Scored,
    Summary,
    summarize,
)
from ember.eval.scorers import Score
from ember.llm import FinishReason, ModelRef, Money, Usage
from ember.llm.request import JsonValue

__all__ = ["SCHEMA_VERSION", "RunWriter", "read_run"]

SCHEMA_VERSION = 1

_Object = dict[str, JsonValue]
_EnumT = TypeVar("_EnumT", FinishReason, ErrorKind)


class RunWriter:
    """Writes one run's records to a new file, flushing every line.

    The file must not exist yet, so a run never overwrites an earlier one.
    Use it as a context manager; it owns the file handle until the block ends.
    """

    def __init__(self, path: Path) -> None:
        self._handle: IO[str] = path.open("x", encoding="utf-8")

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._handle.close()

    def header(self, header: RunHeader) -> None:
        """Writes the first record."""
        self._write("run", _encode_header(header))

    def item(self, record: ItemRecord) -> None:
        """Writes one item record."""
        self._write("item", _encode_item(record))

    def end(self, end: RunEnd, summary: Summary) -> None:
        """Writes the last record."""
        self._write("end", _encode_end(end, summary))

    def _write(self, kind: str, fields: _Object) -> None:
        line = json.dumps({"schema": SCHEMA_VERSION, "type": kind, **fields}, allow_nan=False)
        self._handle.write(line + "\n")
        self._handle.flush()


def read_run(path: Path) -> RunRecords:
    """Reads a run file written by `ember.eval.run`.

    Returns:
        The run's records, with the summary recomputed from the item records.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: A line other than an unterminated last one is not a valid
            record of schema version 1, the records are out of order, or the
            summary written at the end does not match the item records.
    """
    records, partial_dropped = _parse_lines(path)
    if not records:
        raise ValueError(f"{path}: empty run file")
    header = _decode_header(records[0].of_type("run"))
    items: list[ItemRecord] = []
    end: tuple[RunEnd, Summary] | None = None
    for record in records[1:]:
        if end is not None:
            raise ValueError(f"{record.where}: a record follows the end record")
        if record.text("type") == "end":
            end = _decode_end(record)
            continue
        item = _decode_item(record.of_type("item"))
        if item.index != len(items):
            raise ValueError(f"{record.where}: item index {item.index}, expected {len(items)}")
        items.append(item)
    summary = summarize(items)
    if end is not None and partial_dropped:
        raise ValueError(f"{path}: unterminated text after the end record")
    if end is None:
        return RunRecords(header, tuple(items), summary, CutOff(partial_dropped))
    if end[1] != summary:
        raise ValueError(f"{path}: the end record's summary does not match the item records")
    return RunRecords(header, tuple(items), summary, end[0])


def _parse_lines(path: Path) -> tuple[list["_Fields"], bool]:
    """Parses every line; an unterminated last line that is not JSON is dropped."""
    with path.open(encoding="utf-8", newline="") as handle:
        lines = handle.read().split("\n")
    last = lines.pop()  # Empty when the file ends with a newline, as every record does.
    partial_dropped = bool(last.strip()) and not _is_json(last)
    if not partial_dropped:
        lines.append(last)
    records = [
        _Fields.parse(line, f"{path} line {number}")
        for number, line in enumerate(lines, 1)
        if line.strip()
    ]
    return records, partial_dropped


def _is_json(line: str) -> bool:
    try:
        json.loads(line)
    except json.JSONDecodeError:
        return False
    return True


def _encode_header(header: RunHeader) -> _Object:
    return {
        "run_id": header.run_id,
        "started_at": header.started_at,
        "ember_version": header.ember_version,
        "scorer": header.scorer,
        "source": None if header.source is None else _encode_source(header.source),
    }


def _encode_source(source: Source) -> _Object:
    return {
        "path": source.path,
        "config": source.config,
        "split": source.split,
        "revision": source.revision,
        "limit": source.limit,
    }


def _encode_item(record: ItemRecord) -> _Object:
    fields: _Object = {"index": record.index, "latency_s": record.latency_s, "item": record.item}
    if isinstance(record.outcome, Scored):
        fields["outcome"] = "scored"
        fields["output"] = record.outcome.output
        fields["score"] = record.outcome.score.value
        fields["extracted"] = record.outcome.score.extracted
    else:
        fields["outcome"] = "failed"
        fields["error"] = _encode_failed(record.outcome)
    fields["attempts"] = [_encode_attempt(attempt) for attempt in record.attempts]
    return fields


def _encode_attempt(attempt: AttemptRecord) -> _Object:
    fields: _Object = {
        "call_id": attempt.call_id,
        "number": attempt.number,
        "model": str(attempt.model),
        "latency_s": attempt.latency_s,
        "request_id": attempt.request_id,
        "will_retry": attempt.will_retry,
    }
    outcome = attempt.outcome
    if isinstance(outcome, Completed):
        fields["outcome"] = "completed"
        fields["finish_reason"] = outcome.finish_reason.value
        fields["usage"] = _encode_usage(outcome.usage)
        fields["cost_nanos"] = None if outcome.cost is None else outcome.cost.nanos
    else:
        fields["outcome"] = "failed"
        fields["error"] = _encode_failed(outcome)
    return fields


def _encode_failed(failed: Failed) -> _Object:
    return {
        "type": failed.error_type,
        "kind": None if failed.kind is None else failed.kind.value,
        "status": failed.status,
        "message": failed.message,
    }


def _encode_usage(usage: Usage) -> _Object:
    return {
        "input": usage.input,
        "cache_read": usage.cache_read,
        "cache_write": usage.cache_write,
        "output": usage.output,
        "reasoning": usage.reasoning,
        "measured": usage.measured,
    }


def _encode_end(end: RunEnd, summary: Summary) -> _Object:
    abort: JsonValue = None
    if end.abort is not None:
        abort = {"index": end.abort.index, "error": _encode_failed(end.abort.error)}
    return {"elapsed_s": end.elapsed_s, "abort": abort, "summary": _encode_summary(summary)}


def _encode_summary(summary: Summary) -> _Object:
    return {
        "items": summary.items,
        "scored": summary.scored,
        "failed": summary.failed,
        "mean_score": summary.mean_score,
        "failures": dict(summary.failures),
        "calls": summary.calls,
        "attempts": summary.attempts,
        "retries": summary.retries,
        "failed_attempts": summary.failed_attempts,
        "failed_calls": summary.failed_calls,
        "unpriced_calls": summary.unpriced_calls,
        "unmeasured_calls": summary.unmeasured_calls,
        "cost_nanos": summary.cost.nanos,
        "usage": _encode_usage(summary.usage),
    }


class _Fields:
    """Typed access to one decoded JSON object; every miss raises naming the line."""

    def __init__(self, data: Mapping[str, JsonValue], where: str) -> None:
        self.data = data
        self.where = where

    @classmethod
    def parse(cls, line: str, where: str) -> "_Fields":
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where}: not JSON ({exc.msg})") from exc
        fields = cls.of(data, where)
        if fields.integer("schema") != SCHEMA_VERSION:
            raise ValueError(f"{where}: schema {fields.data['schema']} is not {SCHEMA_VERSION}")
        return fields

    @classmethod
    def of(cls, data: object, where: str) -> "_Fields":
        if not isinstance(data, dict):
            raise ValueError(f"{where}: expected a JSON object")
        return cls(data, where)

    def of_type(self, kind: str) -> "_Fields":
        if self.text("type") != kind:
            raise ValueError(f"{self.where}: expected a {kind!r} record")
        return self

    def _get(self, key: str) -> JsonValue:
        if key not in self.data:
            raise ValueError(f"{self.where}: missing {key!r}")
        return self.data[key]

    def _fail(self, key: str, expected: str) -> ValueError:
        return ValueError(f"{self.where}: {key!r} must be {expected}")

    def json(self, key: str) -> JsonValue:
        return self._get(key)

    def text(self, key: str) -> str:
        value = self._get(key)
        if not isinstance(value, str):
            raise self._fail(key, "a string")
        return value

    def choice(self, key: str, allowed: tuple[str, ...]) -> str:
        value = self.text(key)
        if value not in allowed:
            raise ValueError(f"{self.where}: {key!r} has unknown value {value!r}")
        return value

    def optional_text(self, key: str) -> str | None:
        return None if self._get(key) is None else self.text(key)

    def integer(self, key: str) -> int:
        value = self._get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise self._fail(key, "an integer")
        return value

    def optional_integer(self, key: str) -> int | None:
        return None if self._get(key) is None else self.integer(key)

    def number(self, key: str) -> float:
        value = self._get(key)
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise self._fail(key, "a number")
        return float(value)

    def optional_number(self, key: str) -> float | None:
        return None if self._get(key) is None else self.number(key)

    def flag(self, key: str) -> bool:
        value = self._get(key)
        if not isinstance(value, bool):
            raise self._fail(key, "true or false")
        return value

    def nested(self, key: str) -> "_Fields":
        return _Fields.of(self._get(key), f"{self.where} {key}")

    def optional_nested(self, key: str) -> "_Fields | None":
        return None if self._get(key) is None else self.nested(key)

    def nested_list(self, key: str) -> list["_Fields"]:
        value = self._get(key)
        if not isinstance(value, list):
            raise self._fail(key, "a list")
        return [_Fields.of(entry, f"{self.where} {key}[{i}]") for i, entry in enumerate(value)]

    def counts(self, key: str) -> dict[str, int]:
        nested = self.nested(key)
        return {name: nested.integer(name) for name in nested.data}


def _decode_header(fields: _Fields) -> RunHeader:
    source = fields.optional_nested("source")
    return RunHeader(
        run_id=fields.text("run_id"),
        started_at=fields.text("started_at"),
        ember_version=fields.text("ember_version"),
        scorer=fields.text("scorer"),
        source=None if source is None else _decode_source(source),
    )


def _decode_source(fields: _Fields) -> Source:
    return Source(
        path=fields.text("path"),
        config=fields.optional_text("config"),
        split=fields.text("split"),
        revision=fields.optional_text("revision"),
        limit=fields.optional_integer("limit"),
    )


def _decode_item(fields: _Fields) -> ItemRecord:
    outcome: Scored | Failed
    if fields.choice("outcome", ("scored", "failed")) == "scored":
        score = Score(fields.number("score"), fields.optional_text("extracted"))
        outcome = Scored(fields.json("output"), score)
    else:
        outcome = _decode_failed(fields.nested("error"))
    return ItemRecord(
        index=fields.integer("index"),
        item=fields.json("item"),
        latency_s=fields.number("latency_s"),
        outcome=outcome,
        attempts=tuple(_decode_attempt(entry) for entry in fields.nested_list("attempts")),
    )


def _decode_attempt(fields: _Fields) -> AttemptRecord:
    outcome: Completed | Failed
    if fields.choice("outcome", ("completed", "failed")) == "completed":
        cost = fields.optional_integer("cost_nanos")
        outcome = Completed(
            finish_reason=_member(FinishReason, fields, "finish_reason"),
            usage=_decode_usage(fields.nested("usage")),
            cost=None if cost is None else Money(cost),
        )
    else:
        outcome = _decode_failed(fields.nested("error"))
    return AttemptRecord(
        call_id=fields.text("call_id"),
        number=fields.integer("number"),
        model=ModelRef.parse(fields.text("model")),
        latency_s=fields.number("latency_s"),
        request_id=fields.optional_text("request_id"),
        will_retry=fields.flag("will_retry"),
        outcome=outcome,
    )


def _decode_failed(fields: _Fields) -> Failed:
    kind = fields.optional_text("kind")
    return Failed(
        error_type=fields.text("type"),
        kind=None if kind is None else _member(ErrorKind, fields, "kind"),
        status=fields.optional_integer("status"),
        message=fields.text("message"),
    )


def _decode_usage(fields: _Fields) -> Usage:
    return Usage(
        input=fields.integer("input"),
        cache_read=fields.integer("cache_read"),
        cache_write=fields.integer("cache_write"),
        output=fields.integer("output"),
        reasoning=fields.integer("reasoning"),
        measured=fields.flag("measured"),
    )


def _decode_end(fields: _Fields) -> tuple[RunEnd, Summary]:
    abort_fields = fields.optional_nested("abort")
    abort = None
    if abort_fields is not None:
        abort = Abort(abort_fields.integer("index"), _decode_failed(abort_fields.nested("error")))
    return RunEnd(fields.number("elapsed_s"), abort), _decode_summary(fields.nested("summary"))


def _decode_summary(fields: _Fields) -> Summary:
    return Summary(
        items=fields.integer("items"),
        scored=fields.integer("scored"),
        failed=fields.integer("failed"),
        mean_score=fields.optional_number("mean_score"),
        failures=MappingProxyType(fields.counts("failures")),
        calls=fields.integer("calls"),
        attempts=fields.integer("attempts"),
        retries=fields.integer("retries"),
        failed_attempts=fields.integer("failed_attempts"),
        failed_calls=fields.integer("failed_calls"),
        unpriced_calls=fields.integer("unpriced_calls"),
        unmeasured_calls=fields.integer("unmeasured_calls"),
        cost=Money(fields.integer("cost_nanos")),
        usage=_decode_usage(fields.nested("usage")),
    )


def _member(enum: type[_EnumT], fields: _Fields, key: str) -> _EnumT:
    value = fields.text(key)
    try:
        return enum(value)
    except ValueError:
        raise ValueError(f"{fields.where}: {key!r} has unknown value {value!r}") from None
