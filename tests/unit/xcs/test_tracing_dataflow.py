"""Regression tests for XCS tracing and dataflow integrity."""

from __future__ import annotations

from ember.xcs.compiler.builder import IRBuilder
from ember.xcs.compiler.parallelism import ParallelismAnalyzer
from ember.xcs.config import Config
from ember.xcs.runtime.engine import ExecutionEngine


def test_ir_builder_produces_edges_and_order() -> None:
    builder = IRBuilder()

    def g(x: int) -> int:
        return x * 2

    def h(x: int) -> int:
        y = g(x)
        return y + 1

    graph = builder.trace(h, 3)
    name_to_node = {node.operator.__name__: node_id for node_id, node in graph.nodes.items()}

    assert "g" in name_to_node and "h" in name_to_node
    g_id = name_to_node["g"]
    h_id = name_to_node["h"]

    assert h_id in graph.edges.get(g_id, ())
    order = graph.topological_order()
    assert order.index(g_id) < order.index(h_id)


def test_execution_engine_uses_live_outputs() -> None:
    builder = IRBuilder()

    def g(x: int) -> int:
        return x * 2

    def h(x: int) -> int:
        return g(x) + 1

    # Build graph using a different input than execution to ensure no value replay.
    graph = builder.trace(h, 2)
    analysis = ParallelismAnalyzer().analyze(graph)
    engine = ExecutionEngine()

    result = engine.execute(graph, args=(5,), kwargs={}, parallelism=analysis, config=Config())
    assert result == 11


def test_parallel_groups_respect_dependencies() -> None:
    builder = IRBuilder()

    def g(x: int) -> int:
        return x * 2

    def h(x: int) -> int:
        return g(x) + 1

    graph = builder.trace(h, 3)
    name_to_node = {node.operator.__name__: node_id for node_id, node in graph.nodes.items()}
    g_id = name_to_node["g"]
    h_id = name_to_node["h"]
    assert graph.edges.get(g_id) == (h_id,)
    analysis = ParallelismAnalyzer().analyze(graph)

    assert analysis.parallel_groups == []


def test_nondeterministic_results_not_replayed() -> None:
    counter = {"value": 0}

    def g(_: int) -> int:
        counter["value"] += 1
        return counter["value"]

    def h(x: int) -> int:
        return g(x)

    builder = IRBuilder()
    graph = builder.trace(h, 0)
    analysis = ParallelismAnalyzer().analyze(graph)
    engine = ExecutionEngine()

    first = engine.execute(graph, args=(0,), kwargs={}, parallelism=analysis, config=Config())
    second = engine.execute(graph, args=(0,), kwargs={}, parallelism=analysis, config=Config())

    assert first != second and second > first
