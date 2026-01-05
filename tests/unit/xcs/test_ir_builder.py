"""Regression tests for the IR builder + execution engine handshake."""

from __future__ import annotations

import threading

import pytest

from ember.api.decorators import mark_orchestration
from ember.xcs.compiler.builder import IRBuilder
from ember.xcs.runtime.engine import ExecutionEngine


@pytest.fixture
def builder() -> IRBuilder:
    return IRBuilder()


@pytest.fixture
def engine() -> ExecutionEngine:
    inst = ExecutionEngine(max_workers=1)
    try:
        yield inst
    finally:
        inst.shutdown()


def test_trace_emits_callable_node_with_kwarg_specs(builder: IRBuilder) -> None:
    def greet(name: str, *, excited: bool = False) -> str:
        return f"Hi {name}!" if excited else f"Hi {name}."

    graph = builder.trace(greet, "Ada", excited=True)
    assert "return" not in graph.nodes
    assert graph.nodes

    node = next(iter(graph.nodes.values()))
    assert node.operator is greet
    assert node.metadata.get("is_return") is True
    assert node.outputs
    assert node.outputs[0] != "return_value"
    call_spec_kw = node.metadata.get("call_spec_kw", {})
    assert call_spec_kw["excited"]["kind"] == "kwarg"
    assert call_spec_kw["excited"]["name"] == "excited"


def test_execution_engine_replays_with_fresh_args(
    builder: IRBuilder, engine: ExecutionEngine
) -> None:
    def add(x: int, y: int) -> int:
        return x + y

    graph = builder.trace(add, 1, 2)
    result = engine.execute(graph, (10, 5), {})
    assert result == 15


def test_execution_engine_respects_callable_kwargs(
    builder: IRBuilder, engine: ExecutionEngine
) -> None:
    def greet(name: str, *, excited: bool = False) -> str:
        return f"Hi {name}!" if excited else f"Hi {name}."

    graph = builder.trace(greet, "Ada", excited=True)
    quiet_result = engine.execute(graph, ("Tess",), {"excited": False})
    loud_result = engine.execute(graph, ("Lin",), {"excited": True})
    assert quiet_result == "Hi Tess."
    assert loud_result == "Hi Lin!"


def test_tuple_return_round_trips(builder: IRBuilder, engine: ExecutionEngine) -> None:
    def pair(x: int) -> tuple[int, int]:
        return x, x + 1

    graph = builder.trace(pair, 1)
    result = engine.execute(graph, (4,), {})
    assert result == (4, 5)


def test_trace_handles_lambda_with_closure(builder: IRBuilder) -> None:
    def outer(value: int) -> int:
        scale = 2
        return (lambda inner: inner + scale)(value)

    graph = builder.trace(outer, 10)
    operators = [node.operator for node in graph.nodes.values()]
    assert any(getattr(op, "__name__", "") == "<lambda>" for op in operators)


def test_builder_allows_concurrent_traces(builder: IRBuilder) -> None:
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    graphs = []

    def worker(seed: int) -> None:
        def compute(x: int) -> int:
            return x + seed

        try:
            barrier.wait()
            graph = builder.trace(compute, seed)
            graphs.append(graph)
        except BaseException as exc:  # pragma: no cover - diagnostic
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(idx,)) for idx in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(graphs) == 2
    for graph in graphs:
        assert graph.nodes


def test_mark_orchestration_sets_metadata(builder: IRBuilder) -> None:
    @mark_orchestration
    def orchestrate(value: int) -> int:
        return value + 1

    graph = builder.trace(orchestrate, 3)
    orchestration_flags = [node.metadata.get("is_orchestration") for node in graph.nodes.values()]
    assert any(orchestration_flags)
