"""Unit tests for XCSNoOpScheduler.

This module tests sequential execution using the XCSNoOpScheduler.
"""

from typing import Any, Dict

from ember.xcs.engine.xcs_noop_scheduler import XCSNoOpScheduler
from ember.xcs.engine.xcs_engine import compile_graph
from ember.xcs.graph.xcs_graph import XCSGraph


def dummy_operator(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """A simple operator that multiplies input 'value' by 2."""
    return {"out": inputs["value"] * 2}


def test_noop_scheduler() -> None:
    """Tests that XCSNoOpScheduler runs tasks sequentially."""
    graph = XCSGraph()
    graph.add_node(operator=dummy_operator, node_id="node1")
    plan = compile_graph(graph=graph)
    scheduler = XCSNoOpScheduler()
    results = scheduler.run_plan(plan=plan, global_input={"value": 3}, graph=graph)
    assert results["node1"] == {"out": 6}
