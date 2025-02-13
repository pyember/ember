"""
Unified XCS Execution Engine.

This module provides:
  - XCSPlanTask, XCSPlan, and XCSScheduler for executing an XCSGraph.
  - The execute_graph() function as the single entry point for graph execution.
"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Final

from ..graph.xcs_graph import XCSGraph

# Module-level constants for common keys.
_RESPONSE_KEY: Final[str] = "responses"
_FINAL_ANSWER_KEY: Final[str] = "final_answer"
_QUERY_KEY: Final[str] = "query"
_NODE_ATTRIBUTES_KEY: Final[str] = "node_attributes"

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class XCSPlanTask:
    """Represents a single task in the final execution plan.

    Attributes:
        node_id (str): Identifier of the node in the XCSGraph.
        operator (Callable[..., Any]): The callable operator to be executed.
        inbound_nodes (List[str]): List of node IDs that supply inputs to this task.
    """
    node_id: str
    operator: Callable[..., Any]
    inbound_nodes: List[str] = field(default_factory=list)


class XCSPlan:
    """Represents the execution plan for an entire XCSGraph."""

    def __init__(self) -> None:
        """Initializes an empty XCSPlan."""
        self.tasks: Dict[str, XCSPlanTask] = {}

    def add_task(self, *, task: XCSPlanTask) -> None:
        """Adds a task to the execution plan.

        Args:
            task (XCSPlanTask): The task to add.

        Raises:
            ValueError: If a task with the same node_id already exists.
        """
        if task.node_id in self.tasks:
            raise ValueError(f"Task '{task.node_id}' already exists in the plan.")
        self.tasks[task.node_id] = task


def compile_graph(*, graph: XCSGraph) -> XCSPlan:
    """Compiles an XCSGraph into an execution plan.

    Optimization or operator-fusion passes can be added in the future.

    Args:
        graph (XCSGraph): The XCSGraph instance to compile.

    Returns:
        XCSPlan: The resulting execution plan.
    """
    plan: XCSPlan = XCSPlan()
    for node_id, node in graph.nodes.items():
        task = XCSPlanTask(
            node_id=node_id,
            operator=node.operator,
            inbound_nodes=node.inbound_edges,
        )
        plan.add_task(task=task)
    return plan


class XCSScheduler:
    """Concurrency engine that executes an XCSPlan in dependency order."""

    def __init__(self, *, max_workers: Optional[int] = None) -> None:
        """Initializes the scheduler.

        Args:
            max_workers (Optional[int]): Maximum number of worker threads.
                If None, the executor will decide the default number of threads.
        """
        self.max_workers: Optional[int] = max_workers

    def run_plan(
        self,
        *,
        plan: XCSPlan,
        global_input: Dict[str, Any],
        graph: XCSGraph,
    ) -> Dict[str, Any]:
        """Executes an XCSPlan concurrently in dependency order.

        This method constructs dependency mappings and executes tasks concurrently,
        ensuring that each task runs only after its required inputs become available.

        Args:
            plan (XCSPlan): The execution plan to run.
            global_input (Dict[str, Any]): Global input data used during graph execution.
            graph (XCSGraph): The graph corresponding to the plan.

        Returns:
            Dict[str, Any]: A mapping from node IDs to their computed outputs.

        Raises:
            Exception: Re-raises any exception encountered during task execution.
        """
        # Build dependency count and reverse dependency mappings.
        dependency_count: Dict[str, int] = {}
        reverse_dependencies: Dict[str, List[str]] = {}
        for task_id, task in plan.tasks.items():
            dependency_count[task_id] = len(task.inbound_nodes)
            for parent_id in task.inbound_nodes:
                reverse_dependencies.setdefault(parent_id, []).append(task_id)

        available_tasks: List[str] = [tid for tid, count in dependency_count.items() if count == 0]
        results: Dict[str, Any] = {}
        pending_futures: List[Future[Any]] = []
        future_to_task: Dict[Future[Any], str] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while available_tasks or pending_futures:
                # Submit tasks with all dependencies satisfied.
                while available_tasks:
                    current_task_id: str = available_tasks.pop()
                    input_data: Dict[str, Any] = self._gather_inputs(
                        node_id=current_task_id,
                        results=results,
                        global_input=global_input,
                        graph=graph,
                    )
                    future: Future[Any] = executor.submit(
                        self._exec_operator,
                        node_id=current_task_id,
                        input_data=input_data,
                        graph=graph,
                    )
                    pending_futures.append(future)
                    future_to_task[future] = current_task_id

                # Process completed futures.
                for future in as_completed(list(pending_futures)):
                    pending_futures.remove(future)
                    task_id: str = future_to_task.pop(future)
                    try:
                        output: Any = future.result()
                        results[task_id] = output
                        # Update dependency counts for downstream tasks.
                        for child_id in reverse_dependencies.get(task_id, []):
                            dependency_count[child_id] -= 1
                            if dependency_count[child_id] == 0:
                                available_tasks.append(child_id)
                    except Exception as error:
                        logger.exception("Task %s failed with error: %r", task_id, error)
                        raise
        return results

    def _gather_inputs(
        self,
        *,
        node_id: str,
        results: Dict[str, Any],
        global_input: Dict[str, Any],
        graph: XCSGraph,
    ) -> Dict[str, Any]:
        """Aggregates input data for a node by merging global inputs with upstream outputs.

        Args:
            node_id (str): Identifier for the current node.
            results (Dict[str, Any]): Outputs from previously executed nodes.
            global_input (Dict[str, Any]): Global input data for graph execution.
            graph (XCSGraph): The complete graph.

        Returns:
            Dict[str, Any]: Combined input values for the node.
        """
        node = graph.get_node(node_id=node_id)
        node_operator = node.operator
        required_inputs: List[str] = []
        if hasattr(node_operator, "get_signature"):
            signature = node_operator.get_signature()
            if hasattr(signature, "required_inputs"):
                required_inputs = signature.required_inputs  # type: ignore
        inputs: Dict[str, Any] = dict(global_input)
        for parent_id in node.inbound_edges:
            parent_output: Dict[str, Any] = results.get(parent_id, {})
            if _RESPONSE_KEY in required_inputs:
                responses = parent_output.get(_RESPONSE_KEY)
                if isinstance(responses, list):
                    inputs.setdefault(_RESPONSE_KEY, []).extend(responses)
                elif _FINAL_ANSWER_KEY in parent_output:
                    inputs.setdefault(_RESPONSE_KEY, []).append(parent_output[_FINAL_ANSWER_KEY])
            if _QUERY_KEY in required_inputs and _QUERY_KEY not in inputs:
                if _QUERY_KEY in parent_output:
                    inputs[_QUERY_KEY] = parent_output[_QUERY_KEY]

        if node.attributes:
            inputs[_NODE_ATTRIBUTES_KEY] = node.attributes

        return inputs

    def _exec_operator(
        self,
        *,
        node_id: str,
        input_data: Dict[str, Any],
        graph: XCSGraph,
    ) -> Any:
        """Executes the operator for a node using pre-gathered inputs.

        Args:
            node_id (str): Identifier for the node.
            input_data (Dict[str, Any]): Pre-gathered input data for the operator.
            graph (XCSGraph): The complete graph.

        Returns:
            Any: The output produced by the operator.
        """
        node = graph.get_node(node_id=node_id)
        return node.operator(inputs=input_data)


def execute_graph(
    *,
    graph: XCSGraph,
    global_input: Dict[str, Any],
    max_workers: Optional[int] = None,
) -> Any:
    """Compiles and executes an XCSGraph.

    This function compiles the provided XCSGraph into an execution plan, schedules its tasks
    concurrently, and returns the output associated with the graph's exit node if defined; otherwise,
    it returns a complete mapping of node outputs.

    Args:
        graph (XCSGraph): The graph to execute.
        global_input (Dict[str, Any]): Global input data for executing the graph.
        max_workers (Optional[int]): Maximum number of worker threads to utilize.

    Returns:
        Any: Output from the exit node if defined, or a dictionary mapping node IDs to outputs.
    """
    plan: XCSPlan = compile_graph(graph=graph)
    scheduler: XCSScheduler = XCSScheduler(max_workers=max_workers)
    results: Dict[str, Any] = scheduler.run_plan(
        plan=plan,
        global_input=global_input,
        graph=graph,
    )
    if graph.exit_node:
        return results.get(graph.exit_node)
    return results
