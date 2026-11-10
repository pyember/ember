"""The guarantees of `@xcs.jit` in docs/design/v0.2.md, one test each.

Model calls go through the oracle's `FakeModel`, which reaches the scheduler
through `ember.xcs.runtime` exactly as the client does.
"""

import asyncio
import contextvars
import gc
import sys
import threading
import time
import types
import weakref
from collections.abc import Iterator

import fakemodel
import pytest

from ember import xcs
from ember.errors import ErrorKind, ModelError
from ember.llm import Completion, Request
from ember.xcs import runtime

LATENCY_S = 0.2

CALLER_VALUE: contextvars.ContextVar[str] = contextvars.ContextVar("caller_value", default="unset")


def test_body_and_each_request_run_exactly_once() -> None:
    model = fakemodel.FakeModel(0.01)
    runs = []

    @xcs.jit
    def program(x: str) -> list[str]:
        runs.append(x)
        first, second = model(x), model(x + "!")
        return [first.text, second.text]

    assert program("q") == ["<q>", "<q!>"]
    assert runs == ["q"]
    assert sum(model.requests().values()) == model.reached() == 2


def test_request_uses_the_argument_values_at_the_call_site() -> None:
    model = fakemodel.FakeModel(0.05)

    @xcs.jit
    def program() -> str:
        words = ["a"]
        reply = model(" ".join(words))
        words.append("b")
        return reply.text

    assert program() == "<a>"
    assert list(model.requests()) == [repr(Request.from_prompt("a"))]


def test_independent_calls_overlap_and_a_dependent_call_waits_for_its_input() -> None:
    model = fakemodel.FakeModel(LATENCY_S)

    @xcs.jit
    def program() -> str:
        first, second = model("a"), model("b")
        third = model(first.text + second.text)
        return third.text

    started = time.perf_counter()
    assert program() == "<<a><b>>"
    # The first two were answered at once; the third, which reads them, after.
    assert model.peak_in_flight() == 2
    assert time.perf_counter() - started >= 2 * LATENCY_S


def test_nothing_is_kept_between_calls() -> None:
    model = fakemodel.FakeModel(0.01)

    class Payload:
        pass

    @xcs.jit
    def program(argument: Payload) -> str:
        created = Payload()
        created_refs.append(weakref.ref(created))
        return model(f"{id(argument)} {id(created)}").text

    created_refs: list[weakref.ref[Payload]] = []
    argument = Payload()
    argument_ref = weakref.ref(argument)
    program(argument)
    del argument
    gc.collect()
    assert argument_ref() is None
    assert created_refs[0]() is None
    assert program.graph() is not None


def test_user_code_never_runs_in_parallel() -> None:
    model = fakemodel.FakeModel(0.02)
    running = 0
    most = 0
    threads = set()

    def step(label: str) -> None:
        nonlocal running, most
        running += 1
        most = max(most, running)
        threads.add(threading.get_ident())
        time.sleep(0.001)
        running -= 1

    def item(i: int) -> str:
        step(f"{i} before")
        reply = model(str(i))
        step(f"{i} between")
        text = reply.text
        step(f"{i} after")
        return text

    @xcs.jit
    def program() -> list[str]:
        step("body")
        return xcs.map(item, range(8))

    assert program() == [f"<{i}>" for i in range(8)]
    assert most == 1
    assert threads == {threading.get_ident()}


def test_an_unread_failure_raises_at_the_boundary_with_its_original_type() -> None:
    model = fakemodel.FakeModel(0.02)
    pending: list[Completion] = []

    @xcs.jit
    def program() -> str:
        pending.append(model("fail:a"))
        pending.append(model("fail:b"))
        pending.append(model("ok"))
        return "returned"

    with pytest.raises(ModelError) as raised:
        program()
    assert type(raised.value) is ModelError
    assert raised.value.kind is ErrorKind.RATE_LIMITED
    with pytest.raises(ModelError) as first_read:
        pending[0].result()
    assert first_read.value is raised.value
    assert raised.value.__notes__ == [
        "1 other result also failed and was never read: "
        "ModelError: rate_limited: told to fail model=fake/echo"
    ]
    assert all(completion.done() for completion in pending)


def test_every_pending_call_finishes_before_jit_returns() -> None:
    model = fakemodel.FakeModel(LATENCY_S)
    pending: list[Completion] = []

    @xcs.jit
    def program() -> None:
        pending.extend(model(str(i)) for i in range(3))

    program()
    assert [completion.done() for completion in pending] == [True, True, True]


def test_the_body_error_wins_after_pending_calls_finish() -> None:
    model = fakemodel.FakeModel(LATENCY_S)
    pending: list[Completion] = []

    @xcs.jit
    def program() -> None:
        pending.append(model("slow"))
        pending.append(model("fail:unread"))
        raise ValueError("the body failed")

    with pytest.raises(ValueError, match="the body failed") as raised:
        program()
    assert all(completion.done() for completion in pending)
    assert any("ModelError" in note for note in raised.value.__notes__)


def test_a_failure_read_and_caught_is_not_raised_again() -> None:
    model = fakemodel.FakeModel(0.01)

    @xcs.jit
    def program() -> str:
        try:
            return model("fail:x").text
        except ModelError:
            return "recovered"

    assert program() == "recovered"


def test_a_nested_jit_raises_its_unread_failure_at_its_own_boundary() -> None:
    model = fakemodel.FakeModel(0.01)

    @xcs.jit
    def inner() -> str:
        model("fail:inner")
        return "inner returned"

    @xcs.jit
    def outer() -> str:
        try:
            return inner()
        except ModelError:
            return "outer caught it"

    assert outer() == "outer caught it"


def test_the_call_site_context_reaches_greenlets_and_worker_tasks() -> None:
    seen: dict[str, str] = {}

    def served(label: str) -> str:
        seen[label] = CALLER_VALUE.get()
        return label

    def item(i: int) -> str:
        if i == 1:
            CALLER_VALUE.set("item 1")
        seen[f"item {i}"] = CALLER_VALUE.get()
        return runtime.wait(runtime.dispatch(lambda: served(f"task {i}"), label="t"))

    @xcs.jit
    def program() -> list[str]:
        seen["body"] = CALLER_VALUE.get()
        results = xcs.map(item, range(3))
        seen["body after map"] = CALLER_VALUE.get()
        return results

    token = CALLER_VALUE.set("caller")
    try:
        assert program() == ["task 0", "task 1", "task 2"]
    finally:
        CALLER_VALUE.reset(token)
    assert seen == {
        "body": "caller",
        "item 0": "caller",
        "item 1": "item 1",
        "item 2": "caller",
        "task 0": "caller",
        "task 1": "item 1",
        "task 2": "caller",
        "body after map": "caller",
    }


def test_the_body_runs_in_the_callers_own_context() -> None:
    @xcs.jit
    def program() -> None:
        CALLER_VALUE.set("set by the body")

    def caller() -> str:
        program()
        return CALLER_VALUE.get()

    assert contextvars.copy_context().run(caller) == "set by the body"


def test_deferred_is_true_exactly_inside_jit_and_map() -> None:
    seen: dict[str, bool] = {}

    @xcs.jit
    def program() -> None:
        seen["body"] = runtime.deferred()
        xcs.map(lambda _: seen.setdefault("item", runtime.deferred()), [0])
        runtime.wait(
            runtime.dispatch(lambda: seen.setdefault("task", runtime.deferred()), label="t")
        )
        thread = threading.Thread(target=lambda: seen.setdefault("own thread", runtime.deferred()))
        thread.start()
        thread.join()

    seen["outside"] = runtime.deferred()
    program()
    assert seen == {
        "outside": False,
        "body": True,
        "item": True,
        "task": False,
        "own thread": False,
    }


def test_jit_rejects_functions_whose_bodies_run_after_the_call() -> None:
    async def coroutine() -> None:
        pass

    def generator() -> Iterator[int]:
        yield 1

    with pytest.raises(TypeError, match="async"):
        xcs.jit(coroutine)
    with pytest.raises(TypeError, match="generator"):
        xcs.jit(generator)


def test_jit_leaves_the_trace_hook_alone() -> None:
    called: list[str] = []

    def tracer(frame: types.FrameType, event: str, arg: object) -> None:
        if event == "call":
            called.append(frame.f_code.co_name)

    def marker() -> None:
        pass

    @xcs.jit
    def program() -> object:
        marker()
        return sys.gettrace()

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        hook_inside = program()
        hook_after = sys.gettrace()
    finally:
        sys.settrace(previous)
    assert hook_inside is tracer
    assert hook_after is tracer
    assert "marker" in called


def test_jit_binds_as_a_method() -> None:
    model = fakemodel.FakeModel(0.01)

    class Agent:
        def __init__(self, name: str) -> None:
            self.name = name

        @xcs.jit
        def ask(self, question: str) -> str:
            return model(f"{self.name}: {question}").text

    assert Agent("a").ask("q") == "<a: q>"
    assert Agent("b").ask("q") == "<b: q>"
    graph = Agent.ask.graph()
    assert graph is not None
    assert [node.label for node in graph.nodes] == ["fake/echo"]


def test_jit_calls_from_many_threads_overlap_independently() -> None:
    # One model per thread, so each count is of one jit call's own requests.
    models = [fakemodel.FakeModel(LATENCY_S) for _ in range(8)]

    @xcs.jit
    def program(model: fakemodel.FakeModel, x: int) -> list[str]:
        replies = [model(f"{x}.{i}") for i in range(3)]
        return [reply.text for reply in replies]

    finished: list[int] = []

    def run(x: int) -> None:
        assert program(models[x], x) == [f"<{x}.{i}>" for i in range(3)]
        finished.append(x)

    threads = [threading.Thread(target=run, args=(x,)) for x in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(finished) == list(range(8))
    assert [model.peak_in_flight() for model in models] == [3] * 8


def test_jit_costs_little_when_the_body_makes_no_calls() -> None:
    def body(session: object, x: int) -> int:
        return x + 1

    jitted = xcs.jit(body)
    session = object()
    calls = 2000
    started = time.perf_counter()
    for x in range(calls):
        jitted(session, x)
    per_call_s = (time.perf_counter() - started) / calls
    assert per_call_s < 200e-6, f"{per_call_s * 1e6:.1f} us per call"


class Interrupt(BaseException):
    """Stands in for KeyboardInterrupt."""


def test_an_interrupt_in_a_map_item_tears_the_scope_down() -> None:
    cleaned_up: list[int] = []

    def item(i: int) -> None:
        try:
            runtime.wait(runtime.dispatch(lambda: time.sleep(0.02 if i == 0 else 0.5), label="s"))
            if i == 0:
                raise Interrupt
        finally:
            cleaned_up.append(i)

    @xcs.jit
    def program() -> None:
        xcs.map(item, range(2))

    with pytest.raises(Interrupt):
        program()
    assert sorted(cleaned_up) == [0, 1]
    assert runtime.deferred() is False
    assert xcs.jit(lambda: xcs.map(lambda x: x, [1, 2]))() == [1, 2]


def test_attributes_of_the_function_do_not_replace_jit_methods() -> None:
    def program() -> str:
        return "ran"

    program.graph = "an attribute of the function"  # type: ignore[attr-defined]
    jitted = xcs.jit(program)
    assert jitted() == "ran"
    assert isinstance(jitted.graph(), xcs.Graph)
    assert jitted.__name__ == "program"
    assert jitted.__wrapped__ is program


def test_code_that_runs_after_the_scope_ends_is_outside_it() -> None:
    seen: list[object] = []
    captured: list[contextvars.Context] = []

    async def main() -> None:
        @xcs.jit
        def handler() -> None:
            captured.append(contextvars.copy_context())
            loop = asyncio.get_running_loop()
            loop.call_soon(lambda: seen.append(runtime.deferred()))
            loop.call_soon(lambda: seen.append(runtime.dispatch(lambda: 1, label="late").done()))

        handler()
        await asyncio.sleep(0.05)

    asyncio.run(main())
    assert seen == [False, True]
    assert captured[0].run(runtime.deferred) is False
