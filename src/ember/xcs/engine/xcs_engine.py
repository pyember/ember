"""
Unified XCS Execution Engine.

This module implements the core execution plan logic, compiling an XCSGraph into an execution plan,
and scheduling tasks concurrently. All functions and classes use strong typing and named parameter
invocation.
"""

import logging
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Final, Union, TypeVar

from ..graph.xcs_graph import XCSGraph
from ember.core.types.xcs_types import XCSNode, XCSGraph, XCSPlan as XCSPlanProtocol

# Type for results from node execution
XCSResult = TypeVar("XCSResult")

logger: logging.Logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Immutable Execution Plan
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class XCSPlanTask:
    """
    Represents a single task in an execution plan.

    Attributes:
        node_id (str): Identifier of the node in the XCSGraph.
        operator (Callable[..., Any]): The callable operator.
        inbound_nodes (List[str]): List of node IDs that supply inputs.
    """

    node_id: str
    operator: Callable[[Dict[str, Any]], Any]
    inbound_nodes: List[str] = field(default_factory=list)


class XCSPlan:
    """
    Immutable execution plan compiled from an XCSGraph.
    """

    def __init__(self, tasks: Dict[str, XCSPlanTask], original_graph: XCSGraph) -> None:
        """
        Initialize an execution plan.

        Args:
            tasks: Dictionary mapping node IDs to plan tasks
            original_graph: The original graph this plan was compiled from
        """
        self._tasks = tasks
        self.original_graph = original_graph

    @property
    def tasks(self) -> Dict[str, XCSPlanTask]:
        """
        Get all tasks in this plan.

        Returns:
            Dictionary mapping node IDs to tasks
        """
        return self._tasks


def compile_graph(*, graph: XCSGraph) -> XCSPlan:
    """
    Compiles an XCSGraph into an immutable XCSPlan.

    Args:
        graph: The XCSGraph to compile.

    Returns:
        The resulting execution plan.
    """
    tasks: Dict[str, XCSPlanTask] = {}
    for node in graph.nodes.values():
        node_id: str = node.node_id

        if node_id in tasks:
            raise ValueError(f"Task '{node_id}' already exists in the plan.")
        task = XCSPlanTask(
            node_id=node_id,
            operator=node.operator,
            inbound_nodes=node.inbound_edges,
        )
        tasks[node_id] = task
    return XCSPlan(tasks=tasks, original_graph=graph)


# ------------------------------------------------------------------------------
# Scheduler Interface and Implementation
# ------------------------------------------------------------------------------


class IScheduler(ABC):
    """
    Abstract scheduler interface for executing an XCSPlan.
    """

    @abstractmethod
    def run_plan(
        self,
        *,
        plan: XCSPlan,
        global_input: Dict[str, Any],
        graph: XCSGraph,
    ) -> Dict[str, Any]:
        """
        Execute the given plan concurrently and return a mapping of node outputs.

        Args:
            plan: Execution plan to run
            global_input: Input data available to all nodes
            graph: Original graph containing node definitions

        Returns:
            Dictionary mapping node IDs to execution results
        """
        pass


class TopologicalSchedulerWithParallelDispatch(IScheduler):
    """
    Scheduler implementation using topological sorting and parallel dispatch.
    """

    def __init__(self, *, max_workers: Optional[int] = None) -> None:
        """
        Initialize the scheduler.

        Args:
            max_workers: Maximum number of concurrent workers. None means auto.
        """
        self._max_workers = max_workers

    def run_plan(
        self,
        *,
        plan: XCSPlan,
        global_input: Dict[str, Any],
        graph: XCSGraph,
    ) -> Dict[str, Any]:
        # Build dependency counts and reverse dependency mapping.
        dependency_count: Dict[str, int] = {}
        reverse_dependencies: Dict[str, List[str]] = {}
        for task_id, task in plan.tasks.items():
            dependency_count[task_id] = len(task.inbound_nodes)
            for parent in task.inbound_nodes:
                reverse_dependencies.setdefault(parent, []).append(task_id)

        # Initialize tasks with zero dependencies.
        available_tasks: List[str] = [
            tid for tid, count in dependency_count.items() if count == 0
        ]
        results: Dict[str, Any] = {}
        pending_futures: List[Future[Any]] = []
        future_to_task: Dict[Future[Any], str] = {}

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            while available_tasks or pending_futures:
                # Submit all currently available tasks.
                while available_tasks:
                    task_id = available_tasks.pop()
                    input_data = self._gather_inputs(
                        node_id=task_id,
                        results=results,
                        global_input=global_input,
                        graph=graph,
                    )
                    future = executor.submit(
                        self._exec_operator,
                        node_id=task_id,
                        input_data=input_data,
                        graph=graph,
                    )
                    pending_futures.append(future)
                    future_to_task[future] = task_id

                # Process tasks as they complete.
                for future in as_completed(pending_futures.copy()):
                    pending_futures.remove(future)
                    task_id = future_to_task.pop(future)
                    try:
                        result = future.result()
                        results[task_id] = result
                        # Mark dependent tasks as available when all dependencies are satisfied.
                        for child in reverse_dependencies.get(task_id, []):
                            dependency_count[child] -= 1
                            if dependency_count[child] == 0:
                                available_tasks.append(child)
                    except Exception as e:
                        logger.exception("Task %s failed: %r", task_id, e)
                        raise e
        return results

    def _gather_inputs(
        self,
        *,
        node_id: str,
        results: Dict[str, Any],
        global_input: Dict[str, Any],
        graph: XCSGraph,
    ) -> Dict[str, Any]:
        """
        Gathers inputs for a node by merging global input with outputs from upstream tasks.

        Args:
            node_id: ID of the node to gather inputs for
            results: Results from previously executed nodes
            global_input: Input data available to all nodes
            graph: Original graph containing node definitions

        Returns:
            Dictionary of input data for the node
        """
        inputs = dict(global_input)
        node = graph.get_node(node_id=node_id)
        for parent in node.inbound_edges:
            parent_output = results.get(parent, {})
            if isinstance(parent_output, dict):
                inputs.update(parent_output)
        if node.attributes:
            inputs["node_attributes"] = node.attributes
        return inputs

    def _exec_operator(
        self,
        *,
        node_id: str,
        input_data: Dict[str, Any],
        graph: XCSGraph,
    ) -> Any:
        """
        Executes the operator for a given node.

        Args:
            node_id: ID of the node to execute
            input_data: Input data for the node
            graph: Original graph containing node definitions

        Returns:
            Result of the operator execution
        """
        node = graph.get_node(node_id=node_id)
        result = node.operator(inputs=input_data)
        # Capture outputs for debugging or downstream retrieval
        node.captured_outputs = result
        return result


# ------------------------------------------------------------------------------
# Top-Level API
# ------------------------------------------------------------------------------


def execute_graph(
    *,
    graph: Union[XCSGraph, XCSPlan],
    global_input: Dict[str, Any],
    scheduler: Optional[IScheduler] = None,
    concurrency: bool = True,
) -> Dict[str, Any]:
    """
    Executes an XCSGraph or XCSPlan.

    If 'graph' has a 'nodes' attribute, it is assumed to be an XCSGraph and is compiled.
    Otherwise, it is assumed to be an XCSPlan, in which case we use its original_graph.

    Args:
        graph: Graph or plan to execute
        global_input: Input data available to all nodes
        scheduler: Optional custom scheduler implementation
        concurrency: Whether to execute nodes concurrently

    Returns:
        Dictionary mapping node IDs to execution results
    """
    if hasattr(graph, "nodes"):
        plan = compile_graph(graph=graph)
        orig_graph = graph
    else:
        plan = graph
        orig_graph = plan.original_graph
    if scheduler is None:
        scheduler = TopologicalSchedulerWithParallelDispatch()
    if concurrency:
        results = scheduler.run_plan(
            plan=plan, global_input=global_input, graph=orig_graph
        )
        return results
    else:
        results: Dict[str, Any] = {}
        # Iterate over task values instead of keys; use node_id and operator.
        for task in plan.tasks.values():
            input_data = scheduler._gather_inputs(
                node_id=task.node_id,
                results=results,
                global_input=global_input,
                graph=orig_graph,
            )
            results[task.node_id] = task.operator(inputs=input_data)
        return results
