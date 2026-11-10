"""ember.eval.run records every item and every attempt, and reads back exactly."""

import json
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import MappingProxyType

import pytest

import ember
from ember import context, data
from ember import eval as ember_eval
from ember.data import Example
from ember.errors import ErrorKind, ModelError, TieError
from ember.eval import Completed, Failed, NumericAnswer, Scored
from ember.llm import (
    Client,
    FinishReason,
    Message,
    ModelRef,
    Money,
    Request,
    Result,
    TextPart,
    Usage,
    catalog,
    using,
)
from ember.llm.fake import Fail, FakeProvider, Reply
from ember.llm.observe import Attempt, Observer, active_observers
from ember.non import build_graph

MODEL = ModelRef("fake", "m1")


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keeps every test, including real-client ones, away from the user's config and keys."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EMBER_CONFIG", str(tmp_path / "config.yaml"))
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    context.reload()
    yield
    context.reload()


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
        message=Message.assistant(text),
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
    assert records.end == ember_eval.CutOff(partial_record_dropped=False)
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


def three_answered_items(tmp_path: Path) -> Path:
    model = ScriptedModel({"0 + 0?": [reply("0")], "1 + 1?": [reply("2")], "2 + 2?": [reply("4")]})
    run(tmp_path, model)
    return tmp_path / "run.jsonl"


def test_a_file_cut_mid_line_drops_only_the_partial_record(tmp_path: Path) -> None:
    # The process died while writing item 1: the file ends inside that line.
    path = three_answered_items(tmp_path)
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:2]) + lines[2][: len(lines[2]) // 2])
    records = ember_eval.read_run(path)
    assert records.end == ember_eval.CutOff(partial_record_dropped=True)
    assert [item.index for item in records.items] == [0]


def test_a_bad_line_before_the_last_is_rejected(tmp_path: Path) -> None:
    path = three_answered_items(tmp_path)
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join([lines[0], lines[1][:20] + "\n", *lines[2:]]))
    with pytest.raises(ValueError, match="line 2: not JSON"):
        ember_eval.read_run(path)


def test_an_unterminated_last_line_that_parses_is_kept(tmp_path: Path) -> None:
    path = three_answered_items(tmp_path)
    path.write_text(path.read_text().rstrip("\n"))
    records = ember_eval.read_run(path)
    assert isinstance(records.end, ember_eval.RunEnd)


@pytest.mark.parametrize(("line", "key"), [(1, "outcome"), (1, "attempts")])
def test_unknown_outcome_values_are_rejected(tmp_path: Path, line: int, key: str) -> None:
    path = three_answered_items(tmp_path)
    lines = path.read_text().splitlines()
    record = json.loads(lines[line])
    if key == "outcome":
        record["outcome"] = "skipped"
    else:
        record["attempts"][0]["outcome"] = "skipped"
    lines[line] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="'outcome' has unknown value 'skipped'"):
        ember_eval.read_run(path)


def test_items_that_finished_out_of_order_read_back_in_index_order(tmp_path: Path) -> None:
    # Records are written as items finish, which under xcs.map need not be index order.
    path = three_answered_items(tmp_path)
    in_order = ember_eval.read_run(path)
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join([lines[0], lines[3], lines[1], lines[2], *lines[4:]]))
    assert ember_eval.read_run(path) == in_order
    assert [item.index for item in in_order.items] == [0, 1, 2]


def test_a_repeated_item_index_is_rejected(tmp_path: Path) -> None:
    path = three_answered_items(tmp_path)
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join([lines[0], lines[1], lines[1], *lines[3:]]))
    with pytest.raises(ValueError, match="line 3: item index 0 appears twice"):
        ember_eval.read_run(path)


def test_a_finished_run_with_a_missing_item_is_rejected(tmp_path: Path) -> None:
    path = three_answered_items(tmp_path)
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join([lines[0], lines[1], *lines[3:]]))
    with pytest.raises(ValueError, match="finished run is missing item 1"):
        ember_eval.read_run(path)


def test_a_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ember_eval.read_run(tmp_path / "absent.jsonl")


class Pending:
    """A result that raises its error on first read, as calls do under xcs.jit and xcs.map."""

    def __init__(self, outcome: str | ModelError) -> None:
        self._outcome = outcome

    @property
    def text(self) -> str:
        if isinstance(self._outcome, ModelError):
            raise self._outcome
        return self._outcome


def test_an_error_raised_on_first_read_fails_only_its_item(tmp_path: Path) -> None:
    outcomes: dict[str, str | ModelError] = {
        "0 + 0?": "0",
        "1 + 1?": error(ErrorKind.RATE_LIMITED, 429),
        "2 + 2?": "4",
    }
    records = ember_eval.run(
        lambda example: Pending(outcomes[example.question]),
        ITEMS,
        scorer=lambda example, output: NumericAnswer()(example, output.text),
        encode_output=lambda output: output.text,
        records=tmp_path / "run.jsonl",
    )
    assert (records.summary.scored, records.summary.failed) == (2, 1)
    assert dict(records.summary.failures) == {"ModelError:rate_limited": 1}


def test_an_item_that_cannot_be_recorded_is_never_sent(tmp_path: Path) -> None:
    sent: list[object] = []

    def task(item: object) -> str:
        sent.append(item)
        return "x"

    with pytest.raises(TypeError, match="encode_item"):
        ember_eval.run(
            task,
            [object()],
            scorer=lambda i, o: ember_eval.Score(0.0, None),
            records=tmp_path / "run.jsonl",
        )
    assert sent == []


def test_records_hold_json_types_whatever_the_encoder_returns(tmp_path: Path) -> None:
    records = ember_eval.run(
        lambda example: example.answer,
        ITEMS[:1],
        scorer=NumericAnswer(),
        encode_item=lambda example: MappingProxyType({"q": example.question, "pair": (1, 2)}),
        encode_output=lambda output: (output,),
        records=tmp_path / "run.jsonl",
    )
    assert records.items[0].item == {"q": "0 + 0?", "pair": [1, 2]}
    assert records.items[0].outcome == Scored(["0"], ember_eval.Score(1.0, "0"))
    assert records == ember_eval.read_run(tmp_path / "run.jsonl")


def test_each_item_is_on_disk_before_the_next_starts(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    lines_seen: list[int] = []

    def task(example: Example) -> str:
        lines_seen.append(len(path.read_text().splitlines()))
        return example.answer

    ember_eval.run(task, ITEMS, scorer=NumericAnswer(), records=path)
    assert lines_seen == [1, 2, 3]


def test_the_header_records_where_the_items_came_from(tmp_path: Path) -> None:
    source = data.Source("openai/gsm8k", "main", "test", "740312add88f", 3)
    records = ember_eval.run(
        lambda example: example.answer,
        data.Stream(source, iter(ITEMS)),
        scorer=NumericAnswer(),
        records=tmp_path / "run.jsonl",
    )
    assert records.header.source == source
    assert ember_eval.read_run(tmp_path / "run.jsonl").header.source == source
    plain = ember_eval.run(str, [], scorer=NumericAnswer(), records=tmp_path / "plain.jsonl")
    assert plain.header.source is None


def test_text_after_the_end_record_is_rejected(tmp_path: Path) -> None:
    path = three_answered_items(tmp_path)
    path.write_text(path.read_text() + '{"schema": 1, "type": "it')
    with pytest.raises(ValueError, match="unterminated text after the end record"):
        ember_eval.read_run(path)


def test_a_run_through_the_real_client_records_its_retries_and_prices(tmp_path: Path) -> None:
    usage = Usage(input=10, output=2)
    answers: dict[str, list[Reply | Fail]] = {
        "0 + 0?": [Reply("0", usage=usage, request_id="req-a")],
        "1 + 1?": [
            Fail(ErrorKind.OVERLOADED, status=529),
            Reply("2", usage=usage, request_id="req-b"),
        ],
        "2 + 2?": [Fail(ErrorKind.AUTH, status=401)],
    }

    def respond(model: ModelRef, request: Request) -> Reply | Fail:
        (part,) = request.messages[-1].parts
        assert isinstance(part, TextPart)
        return answers[part.text].pop(0)

    fake = FakeProvider(respond=respond)
    with Client(provider_factory=fake.factory, sleep=lambda s: None) as client, using(client):
        records = ember_eval.run(
            lambda example: ember.models("gpt-5.5", example.question).text,
            ITEMS,
            scorer=NumericAnswer(),
            records=tmp_path / "run.jsonl",
        )

    assert records == ember_eval.read_run(tmp_path / "run.jsonl")
    spec = catalog.default().get("openai/gpt-5.5")
    assert spec is not None and spec.rates is not None
    price = spec.rates.cost(usage)
    assert price is not None
    first, retried, failed = (item.attempts for item in records.items)
    assert [(a.request_id, a.outcome) for a in first] == [
        ("req-a", Completed(FinishReason.STOP, usage, price))
    ]
    assert [(a.number, a.will_retry, a.request_id) for a in retried] == [
        (1, True, None),
        (2, False, "req-b"),
    ]
    assert isinstance(retried[0].outcome, Failed)
    assert (retried[0].outcome.kind, retried[0].outcome.status) == (ErrorKind.OVERLOADED, 529)
    assert len({a.call_id for a in retried}) == 1
    assert [(a.number, a.will_retry) for a in failed] == [(1, False)]
    models_sent = {a.model for item in records.items for a in item.attempts}
    assert models_sent == {ModelRef("openai", "gpt-5.5")}
    summary = records.summary
    assert (summary.scored, summary.failed) == (2, 1)
    assert dict(summary.failures) == {"ModelError:auth": 1}
    assert (summary.calls, summary.attempts, summary.retries) == (3, 4, 1)
    assert (summary.failed_attempts, summary.failed_calls) == (2, 1)
    assert (summary.unpriced_calls, summary.cost) == (0, price + price)


def test_a_non_graph_run_records_answers_as_text_and_a_lost_quorum_per_item(
    tmp_path: Path,
) -> None:
    graph = build_graph("2E@openai/gpt-5.5, 1M")

    def respond(model: ModelRef, request: Request) -> Reply | Fail:
        (part,) = request.messages[-1].parts
        assert isinstance(part, TextPart)
        example = next(example for example in ITEMS if example.question in part.text)
        if example.answer == "2":
            return Fail(ErrorKind.SERVER, status=500)
        return Reply(example.answer, usage=Usage(input=10, output=1))

    fake = FakeProvider(respond=respond)
    with Client(provider_factory=fake.factory, sleep=lambda s: None) as client, using(client):
        records = ember_eval.run(
            lambda example: graph(example.question),
            ITEMS,
            scorer=lambda example, answer: NumericAnswer()(example, answer.text),
            records=tmp_path / "run.jsonl",
        )

    assert records == ember_eval.read_run(tmp_path / "run.jsonl")
    first, lost, third = records.items
    assert first.outcome == Scored({"text": "0"}, ember_eval.Score(1.0, "0"))
    assert third.outcome == Scored({"text": "4"}, ember_eval.Score(1.0, "4"))
    assert isinstance(lost.outcome, Failed)
    assert (lost.outcome.error_type, lost.outcome.kind) == ("QuorumError", None)
    assert "quorum not met" in lost.outcome.message
    assert len(lost.attempts) == 6  # Two members, each tried three times.
    assert dict(records.summary.failures) == {"QuorumError": 1}


def test_a_tied_vote_fails_only_its_item(tmp_path: Path) -> None:
    def task(example: Example) -> str:
        if example.answer == "2":
            raise TieError(("2", "3"), 1)
        return example.answer

    records = ember_eval.run(task, ITEMS, scorer=NumericAnswer(), records=tmp_path / "run.jsonl")
    assert (records.summary.scored, dict(records.summary.failures)) == (2, {"TieError": 1})
