"""Unit tests for TopologicalSchedulerWithParallelDispatch.

This module verifies parallel execution using the TopologicalSchedulerWithParallelDispatch scheduler.
"""

from typing import Any, Dict

from ember.xcs.engine.xcs_engine import (
    TopologicalSchedulerWithParallelDispatch,
    compile_graph,
)
from ember.xcs.graph.xcs_graph import XCSGraph


def dummy_operator(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Dummy operator that multiplies the input value by two.

    Args:
        inputs (Dict[str, Any]): A dictionary containing the key 'value' with a numeric value.

    Returns:
        Dict[str, Any]: A dictionary with the key 'out' holding the result of inputs['value'] * 2.
    """
    return {"out": inputs["value"] * 2}


def test_parallel_scheduler() -> None:
    """Tests parallel execution with TopologicalSchedulerWithParallelDispatch.

    This test constructs an XCSGraph with a single node that utilizes the dummy operator to
    double its input value. The graph is compiled into an execution plan, and the scheduler,
    configured with a maximum of 2 workers, concurrently executes the plan. The test then asserts
    that the output for the node identified as 'node1' equals the expected result.

    Returns:
        None
    """
    graph: XCSGraph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="node1")
    plan = compile_graph(graph=graph)
    scheduler: TopologicalSchedulerWithParallelDispatch = (
        TopologicalSchedulerWithParallelDispatch(max_workers=2)
    )
    results: Dict[str, Any] = scheduler.run_plan(
        plan=plan, global_input={"value": 3}, graph=graph
    )
    assert results["node1"] == {"out": 6}
