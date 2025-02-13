from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from ember.xcs.graph.xcs_graph import XCSGraph

GlobalInputT = TypeVar("GlobalInputT")
TaskFunction = Callable[[GlobalInputT, Dict[str, Any]], Any]


@dataclass(frozen=True)
class Task(Generic[GlobalInputT]):
    """Immutable representation of a task in an execution plan.

    Each Task couples a task execution function with optional metadata to provide
    scheduling or concurrency hints during plan execution.

    Attributes:
        function (TaskFunction): A callable that implements the task logic. It accepts a
            global input of type GlobalInputT and a mapping of outputs from upstream tasks,
            returning a task-specific result.
        metadata (Dict[str, Any]): Optional dictionary containing scheduling or concurrency hints.
    """
    function: TaskFunction
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlanBuilder(Generic[GlobalInputT]):
    """Domain-specific language for constructing strongly-typed execution plans.

    This builder enables composition of interdependent tasks that process a given global input.
    Each task is uniquely identified and may consume outputs produced by its dependencies.

    Attributes:
        _tasks (Dict[str, Task[GlobalInputT]]): Mapping of task names to their Task instances.
        _dependencies (Dict[str, List[str]]): Mapping of task names to lists of task names they depend on.
        _task_order (List[str]): List capturing the order in which tasks were added; the final task is
            designated as the exit node.
    """

    def __init__(self) -> None:
        """Initialize a PlanBuilder with empty task and dependency registries."""
        self._tasks: Dict[str, Task[GlobalInputT]] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._task_order: List[str] = []

    def get_task_names(self) -> List[str]:
        """Retrieve the list of task names registered in the execution plan.

        Returns:
            List[str]: A list of unique task names currently defined in the plan.
        """
        return list(self._tasks.keys())

    def add_task(
        self,
        *,
        name: str,
        function: TaskFunction,
        depends_on: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PlanBuilder[GlobalInputT]:
        """Add a new task to the execution plan.

        The provided callable must conform to the following signature:
            function(global_input: GlobalInputT, upstream_outputs: Dict[str, Any]) -> Any

        Args:
            name (str): A unique identifier for the task.
            function (TaskFunction): The callable implementing the task logic.
            depends_on (Optional[List[str]]): List of task names that must complete before this task is executed.
            metadata (Optional[Dict[str, Any]]): Dictionary with optional scheduling or concurrency hints.

        Returns:
            PlanBuilder[GlobalInputT]: The current instance for method chaining.

        Raises:
            ValueError: If a task with the specified name is already defined.
        """
        if name in self._tasks:
            raise ValueError(f"Task '{name}' is already defined.")
        self._tasks[name] = Task(function=function, metadata=metadata or {})
        self._dependencies[name] = depends_on or []
        self._task_order.append(name)
        return self

    def build(self) -> XCSGraph:
        """Construct and return the execution plan as an XCSGraph.

        This method performs the following steps:
            1. Creates a graph node for each task and attaches the corresponding metadata.
            2. Establishes directed edges between nodes based on the declared task dependencies.
            3. Designates the last added task as the exit node of the graph.

        Returns:
            XCSGraph: The fully constructed execution plan graph.

        Raises:
            ValueError: If a declared dependency references an undefined task.
        """
        graph: XCSGraph = XCSGraph()
        name_to_node: Dict[str, str] = {}

        # Step 1: Create graph nodes for all registered tasks.
        for task_name in self._task_order:
            task: Task[GlobalInputT] = self._tasks[task_name]
            node_id: str = graph.add_node(operator=task.function)
            # Propagate scheduling metadata to the corresponding graph node.
            graph.nodes[node_id].attributes.update(task.metadata)
            name_to_node[task_name] = node_id

        # Step 2: Create directed edges based on task dependencies.
        for task_name, dependencies in self._dependencies.items():
            for dependency in dependencies:
                if dependency not in name_to_node:
                    raise ValueError(
                        f"Dependency '{dependency}' not found for task '{task_name}'."
                    )
                graph.add_edge(
                    from_id=name_to_node[dependency],
                    to_id=name_to_node[task_name],
                )

        # Step 3: Designate the last added task as the exit node, if tasks exist.
        if self._task_order:
            final_task: str = self._task_order[-1]
            graph.exit_node = name_to_node[final_task]

        return graph
