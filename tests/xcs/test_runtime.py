"""`ember.xcs.runtime`, the seam between model calls and the scheduler."""

import contextvars
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future

import pytest

from ember import xcs
from ember.errors import ErrorKind, ModelError
from ember.xcs import runtime

LATENCY_S = 0.2
SLACK_S = 0.2

TAG: contextvars.ContextVar[str] = contextvars.ContextVar("tag", default="unset")


class ProviderDown(Exception):
    """A failure raised by a fake request."""


def fail() -> str:
    raise ProviderDown("down")


def test_outside_a_scope_dispatch_runs_on_the_callers_thread_before_returning() -> None:
    caller = threading.get_ident()
    future = runtime.dispatch(threading.get_ident, label="t")
    assert future.done()
    assert runtime.wait(future) == caller


def test_outside_a_scope_wait_raises_the_original_error() -> None:
    future = runtime.dispatch(fail, label="t")
    with pytest.raises(ProviderDown):
        runtime.wait(future)


def test_outside_a_scope_wait_blocks_on_a_future_another_thread_completes() -> None:
    future: Future[str] = Future()
    threading.Timer(0.05, lambda: future.set_result("late")).start()
    assert runtime.wait(future) == "late"


def test_outside_a_scope_dispatch_stream_returns_the_source_itself() -> None:
    def source() -> Iterator[int]:
        yield from range(3)

    assert list(runtime.dispatch_stream(source, label="s")) == [0, 1, 2]


def test_inside_a_scope_tasks_run_on_named_worker_threads_in_the_dispatching_context() -> None:
    def where() -> tuple[str, str]:
        return threading.current_thread().name, TAG.get()

    @xcs.jit
    def program() -> tuple[str, str]:
        TAG.set("dispatching context")
        return runtime.wait(runtime.dispatch(where, label="t"))

    name, tag = contextvars.copy_context().run(program)
    assert name.startswith("ember-xcs-worker-")
    assert tag == "dispatching context"


def test_waiting_on_any_future_lets_other_greenlets_run() -> None:
    permits = [Future[None]() for _ in range(3)]
    threading.Timer(LATENCY_S, lambda: [permit.set_result(None) for permit in permits]).start()

    def item(i: int) -> int:
        runtime.wait(permits[i])
        return i

    started = time.perf_counter()
    assert xcs.map(item, range(3)) == [0, 1, 2]
    assert time.perf_counter() - started <= LATENCY_S + SLACK_S


def test_a_stream_yields_its_items_in_order_then_its_error() -> None:
    def source() -> Iterator[int]:
        yield 1
        yield 2
        raise ModelError(ErrorKind.SERVER, provider="p", model="m", reason="broke mid-stream")

    received: list[int] = []

    @xcs.jit
    def program() -> None:
        for item in runtime.dispatch_stream(source, label="s"):
            received.append(item)

    with pytest.raises(ModelError, match="broke mid-stream"):
        program()
    assert received == [1, 2]


def test_streams_read_in_different_items_overlap() -> None:
    def source(tag: str) -> Iterator[str]:
        for i in range(2):
            time.sleep(LATENCY_S / 2)
            yield f"{tag}{i}"

    def item(tag: str) -> list[str]:
        return list(runtime.dispatch_stream(lambda: source(tag), label="s"))

    started = time.perf_counter()
    assert xcs.map(item, ["a", "b", "c"]) == [["a0", "a1"], ["b0", "b1"], ["c0", "c1"]]
    assert time.perf_counter() - started <= LATENCY_S + SLACK_S


def test_closing_a_stream_early_stops_its_worker() -> None:
    produced: list[int] = []
    closed = threading.Event()

    def source() -> Iterator[int]:
        try:
            for i in range(1000):
                produced.append(i)
                time.sleep(0.005)
                yield i
        finally:
            closed.set()

    @xcs.jit
    def program() -> list[int]:
        stream = runtime.dispatch_stream(source, label="s")
        taken = [next(stream), next(stream)]
        del stream
        return taken

    assert program() == [0, 1]
    assert closed.wait(2.0)
    assert len(produced) < 1000


def test_a_stream_kept_but_unread_raises_its_failure_at_the_jit_boundary() -> None:
    def source() -> Iterator[int]:
        yield 1
        raise ProviderDown("unread")

    kept: list[Iterator[int]] = []

    @xcs.jit
    def program() -> str:
        kept.append(runtime.dispatch_stream(source, label="s"))
        return "returned"

    with pytest.raises(ProviderDown, match="unread"):
        program()


def test_closing_a_stream_before_reading_it_stops_its_worker() -> None:
    pulled: list[int] = []

    def source() -> Iterator[int]:
        for i in range(10):
            pulled.append(i)
            yield i
            time.sleep(0.2)

    @xcs.jit
    def program() -> None:
        runtime.dispatch_stream(source, label="s").close()  # type: ignore[attr-defined]

    started = time.perf_counter()
    program()
    assert time.perf_counter() - started < 0.6
    assert len(pulled) <= 2


@pytest.mark.parametrize("failure_before_drop", [True, False])
def test_dropping_a_stream_abandons_it_whenever_it_fails(failure_before_drop: bool) -> None:
    pulled: list[int] = []

    def source() -> Iterator[int]:
        for i in range(10):
            pulled.append(i)
            if i == 1:
                raise ProviderDown("abandoned")
            yield i
            time.sleep(0.0 if failure_before_drop else 0.2)

    @xcs.jit
    def program() -> str:
        stream = runtime.dispatch_stream(source, label="s")
        if failure_before_drop:
            time.sleep(0.1)
        del stream
        return "returned"

    started = time.perf_counter()
    assert program() == "returned"
    assert time.perf_counter() - started < 0.6
    assert len(pulled) <= 2


def test_a_stream_returned_from_jit_can_be_read_after_it_returns() -> None:
    def source() -> Iterator[int]:
        yield from range(3)

    @xcs.jit
    def program() -> Iterator[int]:
        return runtime.dispatch_stream(source, label="s")

    assert list(program()) == [0, 1, 2]
