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


def test_outside_a_scope_a_stream_is_sent_before_dispatch_stream_returns() -> None:
    log: list[str] = []

    def source() -> Iterator[int]:
        log.append(f"opened on {threading.current_thread().name}")
        yield from range(3)

    stream = runtime.dispatch_stream(source, label="s")
    assert len(log) == 1 and log[0].startswith("opened on ember-xcs-worker-")
    assert list(stream) == [0, 1, 2]


def test_outside_a_scope_a_stream_failure_is_raised_only_when_pulled() -> None:
    def source() -> Iterator[int]:
        raise ProviderDown("before the first item")
        yield 0

    runtime.dispatch_stream(source, label="s")
    stream = runtime.dispatch_stream(source, label="s")
    with pytest.raises(ProviderDown, match="before the first item"):
        next(stream)


def test_outside_a_scope_closing_an_unread_stream_stops_its_worker() -> None:
    pulled: list[int] = []
    closed = threading.Event()

    def source() -> Iterator[int]:
        try:
            for i in range(10):
                pulled.append(i)
                yield i
                time.sleep(0.1)
        finally:
            closed.set()

    runtime.dispatch_stream(source, label="s").close()  # type: ignore[attr-defined]
    assert closed.wait(1.0)
    assert len(pulled) <= 2


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


class Holder:
    """Keeps a reader alive in a reference cycle, collected only by the cycle collector."""

    def __init__(self) -> None:
        self.me = self
        self.stream: Iterator[int] | None = None


@pytest.mark.parametrize("fails_at", ["before its first item", "after one item"])
@pytest.mark.parametrize("disposal", ["dropped", "closed", "held", "held in a cycle"])
def test_an_unread_stream_failure_is_never_raised_however_the_reader_is_left(
    disposal: str, fails_at: str
) -> None:
    pulled: list[int] = []
    kept: list[Iterator[int]] = []

    def source() -> Iterator[int]:
        for i in range(10):
            time.sleep(0.05)
            if fails_at == "before its first item" or i == 1:
                raise ProviderDown("unread")
            pulled.append(i)
            yield i

    @xcs.jit
    def program() -> str:
        stream = runtime.dispatch_stream(source, label="s")
        if disposal == "closed":
            stream.close()  # type: ignore[attr-defined]
        elif disposal == "held":
            kept.append(stream)
        elif disposal == "held in a cycle":
            Holder().stream = stream
        del stream
        return "returned"

    started = time.perf_counter()
    assert program() == "returned"
    assert time.perf_counter() - started < 0.5
    assert len(pulled) <= 1


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


def test_a_stream_returned_from_jit_reads_everything_afterwards() -> None:
    def source() -> Iterator[int]:
        for i in range(3):
            time.sleep(0.05)
            yield i
        raise ProviderDown("after the last item")

    @xcs.jit
    def program() -> Iterator[int]:
        return runtime.dispatch_stream(source, label="s")

    stream = program()
    assert [next(stream) for _ in range(3)] == [0, 1, 2]
    with pytest.raises(ProviderDown, match="after the last item"):
        next(stream)


def test_a_stream_kept_past_the_jit_call_is_drained_before_it_returns() -> None:
    drained = threading.Event()
    kept: list[Iterator[int]] = []

    def source() -> Iterator[int]:
        for i in range(3):
            time.sleep(0.05)
            yield i
        drained.set()

    @xcs.jit
    def program() -> None:
        kept.append(runtime.dispatch_stream(source, label="s"))

    program()
    assert drained.is_set()
    assert list(kept[0]) == [0, 1, 2]


def test_a_stream_read_to_the_end_inside_jit_returns_all_its_items() -> None:
    def source() -> Iterator[int]:
        yield from range(3)

    @xcs.jit
    def program() -> list[int]:
        return list(runtime.dispatch_stream(source, label="s"))

    assert program() == [0, 1, 2]
