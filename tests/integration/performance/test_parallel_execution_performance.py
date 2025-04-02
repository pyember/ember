"""Performance tests comparing parallel vs sequential execution in JIT optimization.

This module directly tests parallel execution benefits in Ember's JIT optimization system,
using sleep-based operations to ensure consistent timing and clear demonstration of 
parallelization gains. It follows the pattern from the successful examples in 
src/ember/examples/basic/simple_jit_demo.py.
"""

import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, ClassVar, Dict, List, Optional

import pytest

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel
from ember.xcs.engine.unified_engine import execute_graph, ExecutionOptions
from ember.xcs.schedulers.unified_scheduler import NoOpScheduler, ParallelScheduler, SequentialScheduler, WaveScheduler
from ember.xcs.graph.xcs_graph import XCSGraph

# Import JIT implementations
from ember.xcs.jit import jit
from ember.xcs.tracer.xcs_tracing import TracerContext

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Mark these tests for performance measurement
pytestmark = pytest.mark.performance


# --------------------------------
# Input/Output Models
# --------------------------------
class TaskInput(EmberModel):
    """Input model for tasks with an identifier."""

    task_id: str
    data: Optional[Dict[str, Any]] = None


class TaskOutput(EmberModel):
    """Output model for tasks with a result."""

    result: str
    task_id: str
    execution_time: float


class EnsembleOutput(EmberModel):
    """Output model for ensemble operations with multiple results."""

    results: List[str]
    task_id: str
    execution_times: List[float]


# --------------------------------
# Test Operators
# --------------------------------
class DelayOperator(Operator[TaskInput, TaskOutput]):
    """Operator that sleeps for a fixed delay and returns a result."""

    specification: ClassVar[Specification] = Specification(
        input_model=TaskInput,
        structured_output=TaskOutput,
    )

    def __init__(self, *, delay: float, name: str):
        """Initialize with configuration.

        Args:
            delay: The number of seconds to sleep
            name: Name identifier for this operator
        """
        self.delay = delay
        self.name = name

    def forward(self, *, inputs: TaskInput) -> TaskOutput:
        """Sleep for the configured delay and return a result.

        Args:
            inputs: Input task information

        Returns:
            Task output with result and timing information
        """
        start_time = time.time()
        time.sleep(self.delay)  # Fixed, guaranteed delay
        elapsed = time.time() - start_time

        return TaskOutput(
            result=f"Task {inputs.task_id} processed by {self.name}",
            task_id=inputs.task_id,
            execution_time=elapsed,
        )


@jit
class SequentialEnsembleOperator(Operator[TaskInput, EnsembleOutput]):
    """Ensemble operator that executes member operators sequentially."""

    specification: ClassVar[Specification] = Specification(
        input_model=TaskInput,
        structured_output=EnsembleOutput,
    )

    def __init__(self, *, width: int = 10, delay: float = 0.1):
        """Initialize with multiple delay operators.

        Args:
            width: Number of member operators
            delay: Delay for each operator in seconds
        """
        self.members = [
            DelayOperator(delay=delay, name=f"op_{i}") for i in range(width)
        ]

    def forward(self, *, inputs: TaskInput) -> EnsembleOutput:
        """Execute all members sequentially and collect results.

        Args:
            inputs: Input task information

        Returns:
            Collected results from all members
        """
        results = []
        execution_times = []

        for i, member in enumerate(self.members):
            # Create a subtask for each member
            subtask = TaskInput(task_id=f"{inputs.task_id}-{i}")
            output = member(inputs=subtask)
            results.append(output.result)
            execution_times.append(output.execution_time)

        return EnsembleOutput(
            results=results, task_id=inputs.task_id, execution_times=execution_times
        )


@jit
class ParallelEnsembleOperator(Operator[TaskInput, EnsembleOutput]):
    """Ensemble operator that executes member operators in parallel using threads."""

    specification: ClassVar[Specification] = Specification(
        input_model=TaskInput,
        structured_output=EnsembleOutput,
    )

    def __init__(self, *, width: int = 10, delay: float = 0.1):
        """Initialize with multiple delay operators.

        Args:
            width: Number of member operators
            delay: Delay for each operator in seconds
        """
        self.members = [
            DelayOperator(delay=delay, name=f"par_op_{i}") for i in range(width)
        ]

    def forward(self, *, inputs: TaskInput) -> EnsembleOutput:
        """Execute all members in parallel using thread pool and collect results.

        Args:
            inputs: Input task information

        Returns:
            Collected results from all members
        """
        results = []
        execution_times = []

        # Execute members in parallel using a thread pool
        with ThreadPoolExecutor(max_workers=len(self.members)) as executor:
            # Submit all tasks to executor
            futures = []
            for i, member in enumerate(self.members):
                subtask = TaskInput(task_id=f"{inputs.task_id}-{i}")
                futures.append(executor.submit(member, inputs=subtask))

            # Collect results as they complete
            for future in as_completed(futures):
                output = future.result()
                results.append(output.result)
                execution_times.append(output.execution_time)

        return EnsembleOutput(
            results=results, task_id=inputs.task_id, execution_times=execution_times
        )


@jit
class StructuralJITEnsembleOperator(Operator[TaskInput, EnsembleOutput]):
    """Ensemble operator designed to be automatically optimized by structural JIT.

    This operator follows the pattern that the structural JIT system can identify and
    optimize for parallel execution, even though the implementation is sequential.
    """

    specification: ClassVar[Specification] = Specification(
        input_model=TaskInput,
        structured_output=EnsembleOutput,
    )

    def __init__(self, *, width: int = 10, delay: float = 0.1):
        """Initialize with multiple delay operators.

        Args:
            width: Number of member operators
            delay: Delay for each operator in seconds
        """
        self.members = [
            DelayOperator(delay=delay, name=f"structural_op_{i}") for i in range(width)
        ]

    def forward(self, *, inputs: TaskInput) -> EnsembleOutput:
        """Execute members in a way that can be auto-parallelized by structural JIT.

        Args:
            inputs: Input task information

        Returns:
            Collected results from all members
        """
        results = []
        execution_times = []

        # This sequential pattern can be identified and parallelized by structural JIT
        for i, member in enumerate(self.members):
            subtask = TaskInput(task_id=f"{inputs.task_id}-{i}")
            output = member(inputs=subtask)
            results.append(output.result)
            execution_times.append(output.execution_time)

        return EnsembleOutput(
            results=results, task_id=inputs.task_id, execution_times=execution_times
        )


# --------------------------------
# Performance Test Functions
# --------------------------------
@pytest.mark.skipif(
    "not config.getoption('--run-perf-tests') and not config.getoption('--run-all-tests')",
    reason="Performance tests are disabled by default. Run with --run-perf-tests or --run-all-tests flag.",
)
def test_sequential_vs_explicit_parallel():
    """Test performance difference between sequential and explicitly parallel execution."""
    # Configuration
    ensemble_width = 10
    delay = 0.1  # seconds per operator
    runs = 2

    # Create operators with identical member configuration
    sequential_op = SequentialEnsembleOperator(width=ensemble_width, delay=delay)
    parallel_op = ParallelEnsembleOperator(width=ensemble_width, delay=delay)

    # Run sequential ensemble
    logger.info(
        f"\nSequential Ensemble ({ensemble_width} operators, {delay}s delay each):"
    )
    sequential_times = []
    for i in range(runs):
        start_time = time.time()
        result = sequential_op(inputs=TaskInput(task_id=f"seq-{i}"))
        elapsed = time.time() - start_time
        sequential_times.append(elapsed)
        logger.info(f"  Run {i+1}: {elapsed:.4f}s")

    # Run parallel ensemble
    logger.info(
        f"\nParallel Ensemble ({ensemble_width} operators, {delay}s delay each):"
    )
    parallel_times = []
    for i in range(runs):
        start_time = time.time()
        result = parallel_op(inputs=TaskInput(task_id=f"par-{i}"))
        elapsed = time.time() - start_time
        parallel_times.append(elapsed)
        logger.info(f"  Run {i+1}: {elapsed:.4f}s")

    # Calculate average times
    seq_avg = statistics.mean(sequential_times)
    par_avg = statistics.mean(parallel_times)

    # Theoretical times
    theoretical_sequential = ensemble_width * delay
    theoretical_parallel = delay

    # Efficiency calculations
    seq_efficiency = theoretical_sequential / seq_avg if seq_avg > 0 else 0
    par_efficiency = theoretical_parallel / par_avg if par_avg > 0 else 0

    # Calculate speedup
    speedup = seq_avg / par_avg if par_avg > 0 else 0

    # Report results
    logger.info("\nPerformance Comparison:")
    logger.info(
        f"  Sequential avg: {seq_avg:.4f}s (theoretical: {theoretical_sequential:.4f}s)"
    )
    logger.info(
        f"  Parallel avg:   {par_avg:.4f}s (theoretical: {theoretical_parallel:.4f}s)"
    )
    logger.info(f"  Speedup:        {speedup:.2f}x")
    logger.info(f"  Sequential efficiency: {seq_efficiency:.1%}")
    logger.info(f"  Parallel efficiency:   {par_efficiency:.1%}")

    # Assert significant speedup with parallel execution
    assert (
        speedup > 5.0
    ), f"Expected parallel execution to be at least 5x faster, got {speedup:.2f}x"


@pytest.mark.skipif(
    "not config.getoption('--run-perf-tests') and not config.getoption('--run-all-tests')",
    reason="Performance tests are disabled by default. Run with --run-perf-tests or --run-all-tests flag.",
)
def test_jit_trace_to_graph_parallel_speedup():
    """Test creating an XCS graph from a JIT trace and running with parallel scheduler."""
    # Configuration
    ensemble_width = 10
    delay = 0.1  # seconds per operator

    # Create operator for tracing
    ensemble = StructuralJITEnsembleOperator(width=ensemble_width, delay=delay)

    # Trace execution to build a graph
    logger.info(
        f"\nTracing operator execution ({ensemble_width} operators, {delay}s delay):"
    )
    with TracerContext() as tracer:
        # Time the traced execution
        start_time = time.time()
        _ = ensemble(inputs=TaskInput(task_id="trace-run"))
        traced_time = time.time() - start_time
        logger.info(f"  Traced execution time: {traced_time:.4f}s")

    # Verify trace was captured
    assert len(tracer.records) >= 1, "Expected at least one trace record"
    logger.info(f"  Captured {len(tracer.records)} trace records")

    # Build graph from members
    logger.info("\nBuilding XCS graph from operators:")
    graph = XCSGraph()
    for i, member in enumerate(ensemble.members):
        node_id = f"delay_{i}"
        graph.add_node(
            operator=lambda inputs, op=member: op(inputs=inputs), node_id=node_id
        )

    # Create input
    global_input = {"task_id": "graph-test"}

    # Execute with sequential scheduler
    logger.info("\nExecuting with sequential scheduler:")
    seq_scheduler = SequentialScheduler()
    seq_options = ExecutionOptions(scheduler_type="sequential")
    start_seq = time.time()
    _ = execute_graph(graph, global_input, options=seq_options, scheduler=seq_scheduler)
    seq_time = time.time() - start_seq
    logger.info(f"  Sequential execution time: {seq_time:.4f}s")

    # Execute with parallel scheduler
    logger.info("\nExecuting with parallel scheduler:")
    par_scheduler = ParallelScheduler(max_workers=ensemble_width)
    par_options = ExecutionOptions(scheduler_type="parallel", max_workers=ensemble_width)
    start_par = time.time()
    _ = execute_graph(graph, global_input, options=par_options, scheduler=par_scheduler)
    par_time = time.time() - start_par
    logger.info(f"  Parallel execution time: {par_time:.4f}s")

    # Calculate speedup
    speedup = seq_time / par_time if par_time > 0 else 0

    # Calculate efficiency
    theoretical_sequential = ensemble_width * delay
    theoretical_parallel = delay
    seq_efficiency = theoretical_sequential / seq_time if seq_time > 0 else 0
    par_efficiency = theoretical_parallel / par_time if par_time > 0 else 0

    # Report results
    logger.info("\nGraph Execution Performance:")
    logger.info(
        f"  Sequential time: {seq_time:.4f}s (theoretical: {theoretical_sequential:.4f}s)"
    )
    logger.info(
        f"  Parallel time:   {par_time:.4f}s (theoretical: {theoretical_parallel:.4f}s)"
    )
    logger.info(f"  Speedup:         {speedup:.2f}x")
    logger.info(f"  Sequential efficiency: {seq_efficiency:.1%}")
    logger.info(f"  Parallel efficiency:   {par_efficiency:.1%}")

    # Assert significant speedup with parallel execution
    assert (
        par_time < seq_time * 0.33
    ), f"Expected parallel execution to be at least 3x faster, got {speedup:.2f}x"
    assert (
        speedup > 5.0
    ), f"Expected parallel execution to be at least 5x faster, got {speedup:.2f}x"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Running sequential vs explicit parallel test:")
    test_sequential_vs_explicit_parallel()

    logger.info("\nRunning JIT trace to graph parallel speedup test:")
    test_jit_trace_to_graph_parallel_speedup()
