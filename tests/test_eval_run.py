"""ember.eval.run records every item and every attempt, and reads back exactly."""

import json
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from ember import eval as ember_eval
from ember.data import Example
from ember.errors import ErrorKind, ModelError
from ember.eval import Completed, Failed, NumericAnswer, Scored
from ember.llm import FinishReason, ModelRef, Money, Result, Usage
from ember.llm.observe import Attempt, Observer, active_observers

MODEL = ModelRef("fake", "m1")
ITEMS = [Example(question=f"{n} + {n}?", answer=str(2 * n)) for n in range(3)]


def reply(text: str, *, cost: int | None = 450, measured: bool = True) -> Result:
    usage = Usage(input=3, output=2) if measured else Usage(measured=False)
    return Result(
        text=text,
        reasoning="",
        tool_calls=(),
        finish_reason=FinishReason.STOP,
        usage=usage,
        cost=None if cost is None else Money(cost),
        model=MODEL,
        request_id=f"req-{text}",
        latency_s=0.01,
    )


def error(kind: ErrorKind, status: int | None = None) -> ModelError:
    return ModelError(
        kind, provider="fake", model="m1", reason="scripted", status=status, request_id="req-err"
    )


class ScriptedModel:
    """Plays scripted attempts per prompt and reports each to the active observers.

    It reports attempts the way the client's contract says: every attempt,
    retries included, to the observers active at the call site, before the
    call returns. With ``from_worker`` it reports from another thread, as the
    client does when a worker thread finishes the attempt.
    """

    def __init__(
        self, script: dict[str, Sequence[Result | ModelError]], from_worker: bool = False
    ) -> None:
        self.script = script
        self.from_worker = from_worker
        self.calls = 0

    def __call__(self, example: Example) -> str:
        self.calls += 1
        call_id = f"call-{self.calls}"
        observers = active_observers()
        outcomes = self.script[example.question]
        for number, outcome in enumerate(outcomes, 1):
            attempt = Attempt(call_id, number, MODEL, 0.01, outcome, number < len(outcomes))
            self._report(observers, attempt)
            if isinstance(outcome, Result):
                return outcome.text
        assert isinstance(outcome, ModelError)
        raise outcome

    def _report(self, observers: tuple[Observer, ...], attempt: Attempt) -> None:
        def deliver() -> None:
            for observer in observers:
                observer.on_attempt(attempt)

        if not self.from_worker:
            deliver()
            return
        worker = threading.Thread(target=deliver)
        worker.start()
        worker.join()


def run(tmp_path: Path, model: ScriptedModel, items: Sequence[Example] = ITEMS):
    return ember_eval.run(model, items, scorer=NumericAnswer(), records=tmp_path / "run.jsonl")


def test_every_item_and_attempt_is_recorded_and_reads_back_exactly(tmp_path: Path) -> None:
    model = ScriptedModel({"0 + 0?": [reply("0")], "1 + 1?": [reply("2")], "2 + 2?": [reply("5")]})
    records = run(tmp_path, model)
    assert records == ember_eval.read_run(tmp_path / "run.jsonl")
    assert [item.index for item in records.items] == [0, 1, 2]
    assert [item.outcome for item in records.items] == [
        Scored("0", ember_eval.Score(1.0, "0")),
        Scored("2", ember_eval.Score(1.0, "2")),
        Scored("5", ember_eval.Score(0.0, "5")),
    ]
    first = records.items[0].attempts
    assert len(first) == 1
    assert first[0].request_id == "req-0"
    assert first[0].outcome == Completed(FinishReason.STOP, Usage(input=3, output=2), Money(450))
    assert records.items[0].item == {
        "question": "0 + 0?",
        "answer": "0",
        "choices": [],
        "metadata": {},
    }
    assert records.end is not None and records.end.abort is None
    assert records.summary.mean_score == pytest.approx(2 / 3)
    assert records.summary.cost == Money(1350)


def test_a_failed_item_is_recorded_and_the_run_goes_on(tmp_path: Path) -> None:
    # v0.1's EvaluationPipeline raised on the first model failure and lost every result.
    model = ScriptedModel(
        {
            "0 + 0?": [reply("0")],
            "1 + 1?": [error(ErrorKind.AUTH, 401)],
            "2 + 2?": [reply("4")],
        }
    )
    records = run(tmp_path, model)
    failed = records.items[1].outcome
    assert failed == Failed("ModelError", ErrorKind.AUTH, 401, str(error(ErrorKind.AUTH, 401)))
    summary = records.summary
    assert (summary.items, summary.scored, summary.failed) == (3, 2, 1)
    assert summary.mean_score == 1.0
    assert dict(summary.failures) == {"ModelError:auth": 1}
    assert (summary.failed_calls, summary.failed_attempts) == (1, 1)
    line = json.loads((tmp_path / "run.jsonl").read_text().splitlines()[2])
    assert line["outcome"] == "failed" and "score" not in line


def test_retries_are_counted_and_attributed_to_their_item(tmp_path: Path) -> None:
    model = ScriptedModel(
        {
            "0 + 0?": [
                error(ErrorKind.RATE_LIMITED, 429),
                error(ErrorKind.SERVER, 500),
                reply("0"),
            ],
            "1 + 1?": [reply("2")],
            "2 + 2?": [reply("4")],
        }
    )
    records = run(tmp_path, model)
    attempts = records.items[0].attempts
    assert [(a.number, a.will_retry) for a in attempts] == [(1, True), (2, True), (3, False)]
    assert [len(item.attempts) for item in records.items] == [3, 1, 1]
    summary = records.summary
    assert (summary.calls, summary.attempts, summary.retries) == (3, 5, 2)
    assert (summary.failed_attempts, summary.failed_calls) == (2, 0)
    assert summary.mean_score == 1.0


def test_unpriced_and_unmeasured_calls_are_counted_not_hidden(tmp_path: Path) -> None:
    model = ScriptedModel(
        {
            "0 + 0?": [reply("0", cost=None)],
            "1 + 1?": [reply("2", measured=False)],
            "2 + 2?": [reply("4")],
        }
    )
    summary = run(tmp_path, model).summary
    assert (summary.unpriced_calls, summary.unmeasured_calls) == (1, 1)
    assert summary.cost == Money(900)
    assert summary.usage == Usage(input=6, output=4)
    assert "unpriced calls 1" in str(summary) and "unmeasured calls 1" in str(summary)


def test_attempts_reported_from_a_worker_thread_are_kept(tmp_path: Path) -> None:
    model = ScriptedModel(
        {"0 + 0?": [error(ErrorKind.OVERLOADED, 529), reply("0")]}, from_worker=True
    )
    records = run(tmp_path, model, ITEMS[:1])
    assert [a.number for a in records.items[0].attempts] == [1, 2]


def test_an_unexpected_error_stops_the_run_and_is_recorded(tmp_path: Path) -> None:
    def task(example: Example) -> str:
        if example.answer == "2":
            raise KeyError("missing field")
        return example.answer

    with pytest.raises(KeyError):
        ember_eval.run(task, ITEMS, scorer=NumericAnswer(), records=tmp_path / "run.jsonl")
    records = ember_eval.read_run(tmp_path / "run.jsonl")
    assert len(records.items) == 1
    assert records.end is not None and records.end.abort is not None
    assert records.end.abort.index == 1
    assert records.end.abort.error.error_type == "KeyError"


def test_item_errors_choose_which_exceptions_fail_only_their_item(tmp_path: Path) -> None:
    def task(example: Example) -> str:
        if example.answer == "2":
            raise TimeoutError("tool did not answer")
        return example.answer

    records = ember_eval.run(
        task,
        ITEMS,
        scorer=NumericAnswer(),
        records=tmp_path / "run.jsonl",
        item_errors=(ModelError, TimeoutError),
    )
    assert dict(records.summary.failures) == {"TimeoutError": 1}


def test_a_file_cut_off_mid_run_still_reads(tmp_path: Path) -> None:
    model = ScriptedModel({"0 + 0?": [reply("0")], "1 + 1?": [reply("2")], "2 + 2?": [reply("4")]})
    run(tmp_path, model)
    path = tmp_path / "run.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:3]))
    records = ember_eval.read_run(path)
    assert records.end is None
    assert records.summary.items == 2


def test_a_summary_that_disagrees_with_the_items_is_rejected(tmp_path: Path) -> None:
    run(tmp_path, ScriptedModel({"0 + 0?": [reply("0")]}), ITEMS[:1])
    path = tmp_path / "run.jsonl"
    lines = path.read_text().splitlines()
    end = json.loads(lines[-1])
    end["summary"]["retries"] = 5
    path.write_text("\n".join([*lines[:-1], json.dumps(end)]) + "\n")
    with pytest.raises(ValueError, match="does not match the item records"):
        ember_eval.read_run(path)


def test_an_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps({"schema": 2, "type": "run"}) + "\n")
    with pytest.raises(ValueError, match="schema 2 is not 1"):
        ember_eval.read_run(path)


def test_a_run_never_overwrites_earlier_records(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text("earlier run\n")
    with pytest.raises(FileExistsError):
        ember_eval.run(str, ITEMS, scorer=NumericAnswer(), records=path)
    assert path.read_text() == "earlier run\n"


def test_items_of_unknown_types_need_an_encoder(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="pass encode_item= or encode_output="):
        ember_eval.to_json(object())
    with pytest.raises(ValueError, match="JSON has no such number"):
        ember_eval.to_json({"x": float("inf")})
    assert ember_eval.to_json({"a": (1, 2.5, None, True)}) == {"a": [1, 2.5, None, True]}
