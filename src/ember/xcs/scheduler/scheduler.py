import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Any, Callable, Dict, List, Optional

LOGGER: logging.Logger = logging.getLogger(__name__)


class ExecutionTask:
    """Represents a single task to be executed within an execution plan.

    An ExecutionTask encapsulates a callable function along with its required inputs
    and dependencies.
    """

    def __init__(
        self,
        *,
        task_id: str,
        function: Callable[..., Any],
        inputs: Dict[str, Any],
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Initializes an ExecutionTask.

        Args:
            task_id (str): A unique identifier for the task.
            function (Callable[..., Any]): The callable to execute.
            inputs (Dict[str, Any]): A dictionary of keyword arguments to pass to the function.
            dependencies (Optional[List[str]]): A list of task IDs that must complete before executing this task.
        """
        self.task_id: str = task_id
        self.function: Callable[..., Any] = function
        self.inputs: Dict[str, Any] = inputs
        self.dependencies: List[str] = dependencies if dependencies is not None else []

    def run(self) -> Any:
        """Executes the task's function using its inputs.

        Returns:
            Any: The result produced by the function.
        """
        return self.function(**self.inputs)


class ExecutionPlan:
    """Manages a collection of ExecutionTasks for orchestrated execution."""

    def __init__(self) -> None:
        """Constructs an empty ExecutionPlan."""
        self.tasks: Dict[str, ExecutionTask] = {}

    def add_task(self, *, task: ExecutionTask) -> None:
        """Adds an ExecutionTask to the plan.

        Args:
            task (ExecutionTask): The task to be added.

        Raises:
            ValueError: If a task with the same task_id already exists in the plan.
        """
        if task.task_id in self.tasks:
            raise ValueError(f"Task ID '{task.task_id}' already exists in the plan.")
        self.tasks[task.task_id] = task


class Scheduler:
    """Schedules and executes tasks from an ExecutionPlan while respecting dependencies.

    Tasks are executed in parallel batches using a ThreadPoolExecutor. A topological sorting
    on the dependency graph determines the order of execution.
    """

    def __init__(self, *, max_workers: Optional[int] = None) -> None:
        """Initializes the Scheduler.

        Args:
            max_workers (Optional[int]): The maximum number of parallel workers to use. If None,
                the system default is used.
        """
        self.max_workers: Optional[int] = max_workers

    def run_plan(self, *, plan: ExecutionPlan) -> Dict[str, Any]:
        """Executes all tasks in the provided ExecutionPlan in dependency order.

        The method computes the in-degree for each task and executes tasks with zero in-degree
        in parallel. As tasks complete, the in-degree of their dependents is decremented,
        and newly ready tasks are scheduled for execution.

        Args:
            plan (ExecutionPlan): The execution plan containing tasks to run.

        Returns:
            Dict[str, Any]: A mapping from task IDs to their corresponding results.
        """
        # Initialize in-degree counts for each task.
        in_degree: Dict[str, int] = {task_id: 0 for task_id in plan.tasks}
        # Build reverse dependency mapping.
        rev_deps: Dict[str, List[str]] = {task_id: [] for task_id in plan.tasks}
        for task_id, task in plan.tasks.items():
            for dependency in task.dependencies:
                in_degree[task_id] += 1
                rev_deps[dependency].append(task_id)

        ready: deque[str] = deque([tid for tid, deg in in_degree.items() if deg == 0])
        results: Dict[str, Any] = {}

        while ready:
            current_batch: List[str] = list(ready)
            ready.clear()

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task_id: Dict[Future, str] = {}
                for task_id in current_batch:
                    task: ExecutionTask = plan.tasks[task_id]
                    future: Future = executor.submit(task.run)
                    future_to_task_id[future] = task_id

                for future in as_completed(future_to_task_id):
                    task_id: str = future_to_task_id[future]
                    result: Any = future.result()
                    results[task_id] = result
                    # Decrease in-degree for dependent tasks; enqueue if ready.
                    for dependent_task_id in rev_deps[task_id]:
                        in_degree[dependent_task_id] -= 1
                        if in_degree[dependent_task_id] == 0:
                            ready.append(dependent_task_id)

        return results
