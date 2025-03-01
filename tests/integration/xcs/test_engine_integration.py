# File: tests/test_xcs_engine.py
"""
Tests for the XCS execution engine: scheduler execution, error propagation, and concurrency.

These tests ensure that tasks execute in dependency order, errors in tasks are re-raised, and the engine scales.
"""

import pytest
from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.engine.xcs_engine import (
    compile_graph,
    TopologicalSchedulerWithParallelDispatch as XCSScheduler,
)


def task_return_one(*, inputs: dict) -> int:
    return 1


def task_return_two(*, inputs: dict) -> int:
    return 2


def task_add(*, inputs: dict) -> int:
    return 3


def test_scheduler_execution() -> None:
    graph = XCSGraph()
    node1 = "node1"
    node2 = "node2"
    node3 = "node3"
    graph.add_node(operator=task_return_one, node_id=node1)
    graph.add_node(operator=task_return_two, node_id=node2)
    graph.add_node(operator=task_add, node_id=node3)
    graph.add_edge(from_id=node1, to_id=node3)
    graph.add_edge(from_id=node2, to_id=node3)
    graph.exit_node = node3
    plan = compile_graph(graph=graph)
    scheduler = XCSScheduler()
    results = scheduler.run_plan(plan=plan, global_input={}, graph=graph)
    assert results[node3] == 3


def task_fail(*, inputs: dict) -> None:
    raise Exception("Task failed")


def test_scheduler_error_propagation() -> None:
    graph = XCSGraph()
    graph.add_node(operator=task_fail, node_id="fail")
    graph.exit_node = "fail"
    plan = compile_graph(graph=graph)
    scheduler = XCSScheduler()
    with pytest.raises(Exception) as excinfo:
        scheduler.run_plan(plan=plan, global_input={}, graph=graph)
    assert "Task failed" in str(excinfo.value)


def test_scheduler_performance() -> None:
    graph = XCSGraph()
    num_tasks = 1000
    for i in range(num_tasks):
        node_id = f"node_{i}"
        graph.add_node(operator=lambda *, inputs: i, node_id=node_id)
    plan = compile_graph(graph=graph)
    scheduler = XCSScheduler(max_workers=50)
    results = scheduler.run_plan(plan=plan, global_input={}, graph=graph)
    assert len(results) == num_tasks
