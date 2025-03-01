from typing import Any, Dict

from ember.xcs.engine.xcs_engine import XCSPlan
from ember.xcs.graph.xcs_graph import XCSGraph


class XCSNoOpScheduler:
    """
    A single-thread (no concurrency) scheduler for XCS. It runs tasks sequentially.
    """

    def run_plan(
        self, *, plan: XCSPlan, global_input: Dict[str, Any], graph: XCSGraph
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        # Iterate over tasks by node_id; call the operator directly with the provided global input.
        for node_id, task in plan.tasks.items():
            result = task.operator(inputs=global_input)
            results[node_id] = result
        return results
