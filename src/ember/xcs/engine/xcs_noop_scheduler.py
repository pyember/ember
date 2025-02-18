from typing import Any, Dict

class XCSNoOpScheduler:
    """
    A single-thread (no concurrency) scheduler for XCS. It runs tasks sequentially.
    """
    def run_plan(self, plan: Any, global_input: Dict[str, Any], graph: Any) -> Any:
        results = {}
        # If the plan has tasks in a known structure, iterate them in a single pass:
        for task in plan.tasks:
            # Each task is presumably a node in the plan with a 'node.operator' or similar
            node = task.node
            input_data = task.compute_inputs(global_input=global_input, graph=graph)
            results[node] = node.operator(inputs=input_data)
        return results 