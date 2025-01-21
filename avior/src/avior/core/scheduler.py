
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)



class ExecutionTask:
    """Represents a single task to be run in an execution plan.

    An ExecutionTask holds a function, its required inputs, and any dependencies (i.e.,
    other tasks that must complete before this one can begin).
    """

    def __init__(
        self,
        task_id: str,
        function: Callable[..., Any],
        inputs: Dict[str, Any],
        dependencies: Optional[List[str]] = None,
    ):
        """Initializes an ExecutionTask.

        Args:
            task_id: A unique string identifier for this task.
            function: A callable, representing the work to be performed.
            inputs: A dictionary of inputs required by the function.
            dependencies: A list of task IDs that must be completed before this task.
        """
        self.task_id = task_id
        self.function = function
        self.inputs = inputs
        self.dependencies = dependencies or []

    def run(self) -> Any:
        """Executes the task function with the provided inputs.

        Returns:
            The result of the task function.
        """
        return self.function(**self.inputs)


class ExecutionPlan:
    """Holds a collection of ExecutionTasks and manages their creation.

    This plan can be passed to a Scheduler for orchestrated or parallelized execution.
    """

    def __init__(self):
        """Initializes an empty ExecutionPlan."""
        self.tasks: Dict[str, ExecutionTask] = {}

    def add_task(self, task: ExecutionTask):
        """Adds a new ExecutionTask to the plan.

        Args:
            task: The ExecutionTask instance to be added.

        Raises:
            ValueError: If a task with the same task_id already exists in the plan.
        """
        if task.task_id in self.tasks:
            raise ValueError(f"Task ID '{task.task_id}' already exists in the plan.")
        self.tasks[task.task_id] = task


class Scheduler:
    """Coordinates the execution of tasks in an ExecutionPlan using a topological ordering.

    Tasks are executed in batches once their dependencies have all completed. By default,
    it uses Python's ThreadPoolExecutor to parallelize execution across multiple workers.
    """

    def __init__(self, max_workers: Optional[int] = None):
        """Initializes the Scheduler with an optional max worker limit.

        Args:
            max_workers: The maximum number of workers to process tasks in parallel. If None,
                Python will use a default number of workers.
        """
        self.max_workers = max_workers

    def run_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Executes all tasks in the given plan in an order respecting their dependencies.

        A topological sort is used to find tasks that have zero dependencies. These tasks
        are batched and run in parallel. When they finish, tasks that depend on them
        may become ready.

        Args:
            plan: The ExecutionPlan containing tasks to run.

        Returns:
            A dictionary mapping task IDs to their respective results.
        """
        in_degree = {tid: 0 for tid in plan.tasks}
        rev_deps = {tid: [] for tid in plan.tasks}
        for tid, task in plan.tasks.items():
            for dep in task.dependencies:
                in_degree[tid] += 1
                rev_deps[dep].append(tid)

        ready = deque([tid for tid, deg in in_degree.items() if deg == 0])
        results = {}

        while ready:
            current_batch = list(ready)
            ready.clear()

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {}
                for tid in current_batch:
                    task = plan.tasks[tid]
                    future = executor.submit(task.run)
                    future_map[future] = tid

                for future in as_completed(future_map):
                    tid = future_map[future]
                    result = future.result()
                    results[tid] = result
                    for nxt in rev_deps[tid]:
                        in_degree[nxt] -= 1
                        if in_degree[nxt] == 0:
                            ready.append(nxt)

        return results
