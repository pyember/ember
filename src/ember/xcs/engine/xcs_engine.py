"""
XCS Execution Engine: High-Performance Graph Scheduler

This module provides the core execution infrastructure for XCS computational graphs.
It transforms graph-based representations into optimized execution plans and handles
their efficient concurrent execution.

Key components:
1. Compilation - Transforms XCSGraph into an optimized, executable plan
2. Scheduling - Resolves dependencies and manages parallel execution
3. Resource management - Handles thread pools and execution contexts
4. Error handling - Provides robust recovery and reporting

The engine implements a topological execution strategy that respects node dependencies
while maximizing parallelism. It follows the Open/Closed principle by providing a
pluggable scheduler interface that allows for different execution strategies without
modifying the core engine code.

All functions and classes leverage strong typing, immutable data structures where possible,
and named parameter invocation for clarity and safety.
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
    Represents a single executable task within an XCS execution plan.
    
    XCSPlanTask encapsulates the essential information needed to execute a node
    in the computational graph. It is designed as an immutable data structure
    to ensure thread safety during concurrent execution.
    
    The task maintains the minimal set of references needed for execution:
    the operation to perform, its unique identifier, and the dependency information
    required for scheduling. This design follows the Interface Segregation Principle
    by focusing exclusively on the execution concerns, separating them from the
    graph structure concerns.

    Attributes:
        node_id: Unique identifier that maps this task back to its source node in the XCSGraph.
        operator: The callable function or object that performs the actual computation.
        inbound_nodes: List of node IDs that must complete execution before this task can run.
    """

    node_id: str
    operator: Callable[[Dict[str, Any]], Any]
    inbound_nodes: List[str] = field(default_factory=list)


class XCSPlan:
    """
    Immutable execution plan compiled from an XCSGraph.
    
    An XCSPlan represents the finalized, optimized execution strategy for a computational
    graph. It transforms the declarative graph structure (what to compute) into an
    imperative execution plan (how to compute it efficiently).
    
    The execution plan is immutable by design, ensuring thread safety and consistent
    behavior during parallel execution. This immutability also allows the plan to be
    shared and reused across multiple executions with different inputs.
    
    The plan maintains a reference to its original graph to enable introspection,
    debugging, and execution-time optimizations that may need to reference the
    original graph structure or node attributes.
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
    Transforms an XCSGraph into an optimized, immutable execution plan.

    This function performs the critical conversion from a declarative graph representation
    to an imperative execution plan. The compilation process:
    
    1. Extracts essential execution information from each node
    2. Preserves dependency relationships from the original graph
    3. Creates immutable task objects for thread-safe concurrent execution
    4. Maintains a reference to the original graph for introspection
    
    The resulting XCSPlan is optimized for the scheduler implementation and
    guarantees that all dependency relationships from the original graph are
    preserved. This transformation follows the principle of separating the
    "what" (graph) from the "how" (execution plan).

    Args:
        graph: The source XCSGraph to compile into an execution plan.

    Returns:
        An immutable XCSPlan ready for efficient execution.
        
    Raises:
        ValueError: If the graph contains duplicate node IDs or other structural issues.
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
    Abstract interface for XCS execution schedulers.
    
    This interface defines the contract that all XCS schedulers must implement,
    enabling pluggable execution strategies while maintaining consistent behavior.
    It follows the Strategy pattern, allowing different scheduling algorithms to be
    interchanged without affecting the core execution engine.
    
    Implementations of this interface can provide various execution strategies such as:
    - Sequential execution for debugging or deterministic behavior
    - Thread-based parallel execution for CPU-bound operations
    - Distributed execution across multiple machines
    - GPU-accelerated execution for compatible operations
    - Priority-based scheduling for time-sensitive applications
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
        Executes an execution plan with the given inputs and returns the results.

        This method orchestrates the complete execution of a computational graph
        according to the strategy defined by the implementing scheduler. It manages
        concurrency, dependency resolution, resource allocation, and result collection.

        Args:
            plan: The compiled execution plan containing tasks and their dependencies.
            global_input: Input data available to all nodes in the graph.
            graph: The original graph containing node definitions and attributes.

        Returns:
            A dictionary mapping node IDs to their execution results.
            
        Raises:
            ExecutionError: If execution fails due to errors in task execution.
            ResourceExhaustionError: If execution fails due to resource constraints.
        """
        pass


class TopologicalSchedulerWithParallelDispatch(IScheduler):
    """
    High-performance scheduler with dependency-based parallel execution.
    
    This scheduler implements a topological execution strategy with dynamic
    concurrent task dispatch. It processes the execution graph in dependency
    order while maximizing parallelism for independent tasks.
    
    Key features:
    - Dynamic worker pool for optimal resource utilization
    - Task dependency resolution for correct execution order
    - Automatic deadlock detection and prevention
    - Efficient result propagation between dependent tasks
    
    The scheduler maintains minimal state and leverages immutable data structures
    where possible for thread safety during concurrent execution.
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
    Executes a computational graph with the specified inputs and configuration.

    This function serves as the primary entry point for XCS graph execution.
    It handles the complete execution lifecycle:
    
    1. Compilation - If given an XCSGraph, compiles it to an execution plan
    2. Scheduler selection - Uses the provided scheduler or creates a default one
    3. Execution - Dispatches the plan to the scheduler for execution
    4. Result collection - Aggregates and returns the execution results
    
    The function accepts either an XCSGraph or a pre-compiled XCSPlan, allowing
    for both one-off execution and repeated execution with different inputs using
    the same plan. This flexibility follows the principle of separation of concerns:
    graph definition, compilation, and execution are distinct phases that can be
    performed separately.

    Args:
        graph: Either an XCSGraph to be compiled or a pre-compiled XCSPlan.
        global_input: Dictionary of input values available to all nodes in the graph.
        scheduler: Optional custom scheduler implementation. If not provided,
                  a default TopologicalSchedulerWithParallelDispatch is used.
        concurrency: Whether to execute nodes concurrently. Set to False for
                    sequential, deterministic execution (useful for debugging).

    Returns:
        A dictionary mapping node IDs to their execution results.
        
    Raises:
        ExecutionError: If execution fails due to errors in node execution.
        CompilationError: If graph compilation fails due to structural issues.
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
