import pytest
from ember.registry.operators.operator_base import (
    ExecutionTask,
    ExecutionPlan,
    Scheduler,
)


def dummy_task_function(**inputs):
    return inputs["val"] * 2


def test_execution_plan_basic():
    plan = ExecutionPlan()
    t1 = ExecutionTask("t1", dummy_task_function, {"val": 10})
    plan.add_task(t1)
    with pytest.raises(ValueError):
        plan.add_task(t1)  # duplicate task_id


def test_scheduler_no_deps():
    plan = ExecutionPlan()
    for i in range(5):
        plan.add_task(ExecutionTask(f"t{i}", dummy_task_function, {"val": i}))
    scheduler = Scheduler()
    results = scheduler.run_plan(plan)
    for i in range(5):
        assert results[f"t{i}"] == i * 2


@pytest.mark.parametrize(
    "dependency_structure",
    [
        # Chain: t0->t1->t2
        [("t0", []), ("t1", ["t0"]), ("t2", ["t1"])],
        # Fan-in: t2 depends on t0, t1
        [("t0", []), ("t1", []), ("t2", ["t0", "t1"])],
        # Fan-out: t0 no deps, t1,t2 depend on t0
        [("t0", []), ("t1", ["t0"]), ("t2", ["t0"])],
    ],
)
def test_scheduler_various_dependencies(dependency_structure):
    """
    Builds a plan from dependency_structure:
    Each tuple: (task_id, [deps]) and each runs dummy_task_function(val=task_id's index).
    Validate execution order and final results.
    TODO: Implement and verify results.
    """
    pass


def test_scheduler_error_propagation():
    """
    If a task raises an exception, ensure Scheduler surfaces it or fails.
    TODO: Implement a task function that raises and check behember.
    """
    pass


def test_scheduler_large_scale():
    """
    Test with many tasks (e.g. 100) to ensure performance and no deadlocks.
    TODO: Create 100 tasks, run, assert completion.
    """
    pass
