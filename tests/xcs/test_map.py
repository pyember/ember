"""The guarantees of `xcs.map` in docs/design/v0.2.md, one test each."""

import contextvars
import time
from collections.abc import Iterator

import fakemodel
import pytest

from ember import xcs
from ember.errors import ModelError
from ember.xcs import runtime

LATENCY_S = 0.2
SLACK_S = 0.2

ITEM_VALUE: contextvars.ContextVar[str] = contextvars.ContextVar("item_value", default="unset")


def test_results_come_back_in_item_order_whatever_order_calls_finish() -> None:
    def item(delay: float) -> float:
        runtime.wait(runtime.dispatch(lambda: time.sleep(delay), label="sleep"))
        return delay

    delays = [0.15, 0.0, 0.1, 0.05]
    assert xcs.map(item, delays) == delays


def test_an_empty_input_returns_an_empty_list() -> None:
    assert xcs.map(lambda x: x, []) == []


def test_items_can_come_from_any_iterable() -> None:
    def numbers() -> Iterator[int]:
        yield from range(3)

    assert xcs.map(lambda x: x * 2, numbers()) == [0, 2, 4]


def test_items_overlap_their_calls() -> None:
    model = fakemodel.FakeModel(LATENCY_S)
    started = time.perf_counter()
    assert xcs.map(lambda x: model(x).text, ["a", "b", "c", "d"]) == ["<a>", "<b>", "<c>", "<d>"]
    assert time.perf_counter() - started <= LATENCY_S + SLACK_S


def test_items_interleave_only_where_one_reads_a_pending_result() -> None:
    model = fakemodel.FakeModel(0.01)
    log: list[tuple[int, str]] = []

    def item(i: int) -> str:
        texts = []
        for round_ in range(3):
            log.append((i, f"work {round_}"))
            reply = model(f"{i}.{round_}")
            log.append((i, "about to read"))
            texts.append(reply.text)
        log.append((i, "done"))
        return " ".join(texts)

    xcs.map(item, range(4))
    for previous, following in zip(log, log[1:], strict=False):
        if following[0] != previous[0]:
            assert previous[1] in ("about to read", "done"), f"switched after {previous}: {log}"


def test_every_item_starts_before_any_item_continues_past_a_wait() -> None:
    log: list[tuple[int, str]] = []

    def item(i: int) -> int:
        log.append((i, "start"))
        time.sleep(0.01)
        runtime.wait(runtime.dispatch(lambda: time.sleep(0.02), label="short"))
        log.append((i, "resumed"))
        return i

    xcs.map(item, range(5))
    first_resume = min(position for position, (_, event) in enumerate(log) if event == "resumed")
    assert log[:first_resume] == [(i, "start") for i in range(5)]


def test_each_item_runs_in_its_own_copy_of_the_callers_context() -> None:
    def item(i: int) -> str:
        before = ITEM_VALUE.get()
        ITEM_VALUE.set(f"item {i}")
        runtime.wait(runtime.dispatch(lambda: None, label="yield"))
        return f"{before} -> {ITEM_VALUE.get()}"

    token = ITEM_VALUE.set("caller")
    try:
        assert xcs.map(item, range(3)) == [f"caller -> item {i}" for i in range(3)]
        assert ITEM_VALUE.get() == "caller"
    finally:
        ITEM_VALUE.reset(token)


def test_after_a_failure_no_further_items_start() -> None:
    started: list[int] = []

    def item(i: int) -> int:
        started.append(i)
        if i == 1:
            raise ValueError("item 1")
        return i

    with pytest.raises(ValueError, match="item 1"):
        xcs.map(item, range(5))
    assert started == [0, 1]


def test_items_already_started_finish_before_the_error_is_raised() -> None:
    finished: list[int] = []

    def item(i: int) -> int:
        runtime.wait(runtime.dispatch(lambda: time.sleep(0.02 + 0.05 * i), label="sleep"))
        if i == 0:
            raise ValueError("item 0")
        finished.append(i)
        return i

    with pytest.raises(ValueError, match="item 0"):
        xcs.map(item, range(4))
    assert sorted(finished) == [1, 2, 3]


def test_the_earliest_failed_item_wins_whatever_order_they_fail_in() -> None:
    def item(i: int) -> int:
        runtime.wait(runtime.dispatch(lambda: time.sleep(0.2 if i == 0 else 0.05), label="s"))
        raise ValueError(f"item {i}")

    with pytest.raises(ValueError, match="item 0") as raised:
        xcs.map(item, range(3))
    assert raised.value.__notes__ == [
        "2 other items also raised: ValueError: item 1; ValueError: item 2"
    ]


def test_nested_maps_cannot_deadlock() -> None:
    model = fakemodel.FakeModel(0.05)

    def inner(i: int) -> list[str]:
        return xcs.map(lambda j: model(f"{i}.{j}").text, range(16))

    grid = xcs.map(inner, range(16))
    assert grid == [[f"<{i}.{j}>" for j in range(16)] for i in range(16)]


def test_a_map_outside_jit_is_its_own_boundary() -> None:
    model = fakemodel.FakeModel(0.02)

    def item(prompt: str) -> str:
        model(prompt)
        return "not read"

    with pytest.raises(ModelError):
        xcs.map(item, ["ok", "fail:x"])


def test_a_map_inside_jit_can_return_pending_results() -> None:
    model = fakemodel.FakeModel(LATENCY_S)

    @xcs.jit
    def program() -> list[str]:
        pending = xcs.map(model, ["a", "b", "c"])
        return [completion.text for completion in pending]

    started = time.perf_counter()
    assert program() == ["<a>", "<b>", "<c>"]
    assert time.perf_counter() - started <= LATENCY_S + SLACK_S
