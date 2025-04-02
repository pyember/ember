"""Just-In-Time optimization with parallel execution demonstration.

This module demonstrates JIT compilation and parallel execution capabilities
provided by the Ember XCS engine. It showcases how computation graphs
are automatically traced, compiled, and optimized for different execution
strategies, with a focus on concurrent execution benefits.

Key concepts demonstrated:
1. JIT tracing and compilation of operator execution graphs
2. Performance comparison between sequential and parallel execution
3. Practical patterns for building concurrent operator workflows
4. Latency improvements through execution strategy optimization

Example usage:
    uv run python -m ember.examples.basic.simple_jit_demo
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

from ember.xcs import (
    jit, 
    execution_options, 
    ExecutionOptions,
    XCSGraph, 
    TracerContext,
    BaseScheduler,
    execute_graph
)
from ember.xcs.engine.execution_options import get_execution_options
from ember.core.registry.operator.base.operator_base import Operator, Specification
from ember.core.types.ember_model import EmberModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Input/Output model definitions
class DelayInput(EmberModel):
    """Input model for delay-based operators.

    Provides task identification for tracing execution paths through the
    computation graph. Used to correlate results back to their originating
    requests throughout the demonstration.

    Attributes:
        task_id: Unique identifier for correlating the task through execution
    """

    task_id: str = Field(description="Unique identifier for the task execution")


class DelayOutput(EmberModel):
    """Output model from delay-based operations.

    Captures operation results along with the original task identifier for
    correlation and tracing. Preserves the execution context through the
    computation graph.

    Attributes:
        result: The textual representation of the operation result
        task_id: The original task identifier passed through from the input
    """

    result: str = Field(description="Result of the operation processing")
    task_id: str = Field(description="Original task identifier for correlation")


class EnsembleOutput(EmberModel):
    """Output model for ensemble operations combining multiple results.

    Aggregates results from multiple child operations into a single collection
    while preserving the original task context for traceability.

    Attributes:
        results: Collection of result strings from child operations
        task_id: The original task identifier passed through from the input
    """

    results: List[str] = Field(description="Aggregated results from child operations")
    task_id: str = Field(description="Original task identifier for correlation")


# Operator specifications
class DelaySpecification(Specification):
    """Specification for delay-based operators.

    Defines the input/output contract for operations that simulate
    computational latency with predictable performance characteristics.

    Attributes:
        input_model: The expected input model type
        structured_output: The produced output model type
    """

    input_model: Type[EmberModel] = DelayInput
    structured_output: Type[EmberModel] = DelayOutput


class EnsembleSpecification(Specification):
    """Specification for ensemble operators.

    Defines the input/output contract for operations that aggregate
    results from multiple child operations into a consolidated output.

    Attributes:
        input_model: The expected input model type
        structured_output: The produced output model type
    """

    input_model: Type[EmberModel] = DelayInput
    structured_output: Type[EmberModel] = EnsembleOutput


class DelayOperator(Operator[DelayInput, DelayOutput]):
    """Simulates computational latency with a configurable delay.

    Used as a building block for demonstrating execution strategies
    with predictable performance characteristics. Each operator introduces
    a fixed delay to simulate a CPU-bound workload.

    Attributes:
        specification: The operator's input/output contract
        delay_seconds: The duration to sleep during execution
        op_id: Unique identifier for this operator instance
    """

    specification: ClassVar[Specification] = DelaySpecification()

    # Field declarations
    delay_seconds: float
    op_id: str

    def __init__(self, *, delay_seconds: float, op_id: str) -> None:
        """Initializes the delay operator with specified parameters.

        Args:
            delay_seconds: Duration to sleep during operation execution
            op_id: Unique identifier for this operator instance
        """
        self.delay_seconds = delay_seconds
        self.op_id = op_id

    def forward(self, *, inputs: DelayInput) -> DelayOutput:
        """Executes the operation with the configured delay.

        Simulates a computation that takes a predictable amount of time,
        creating a controlled environment for measuring execution strategies.

        Args:
            inputs: Task context information containing task identifier

        Returns:
            Operation result with correlation information
        """
        time.sleep(self.delay_seconds)
        return DelayOutput(
            result=f"Operator {self.op_id} completed task {inputs.task_id}",
            task_id=inputs.task_id,
        )


@jit
class JITEnsembleOperator(Operator[DelayInput, EnsembleOutput]):
    """JIT-optimized operator composing multiple delay operators sequentially.

    When executed, this operator creates a trace that can be compiled
    into an execution graph and optimized for different execution strategies.
    This implementation runs child operators sequentially but creates a
    data flow graph where each operation can be identified as independent.

    Attributes:
        specification: The operator's input/output contract
        operators: Collection of child delay operators to execute
    """

    specification: ClassVar[Specification] = EnsembleSpecification()

    # Field declarations
    operators: List[DelayOperator]

    def __init__(self, *, num_ops: int = 3, delay: float = 0.1) -> None:
        """Initializes the ensemble with a collection of delay operators.

        Creates a sequence of delay operators that will be executed
        according to the current execution context and tracing state.

        Args:
            num_ops: Number of child operators to create
            delay: Delay duration for each child operator in seconds
        """
        self.operators = [
            DelayOperator(delay_seconds=delay, op_id=f"Op-{i+1}")
            for i in range(num_ops)
        ]

    def forward(self, *, inputs: DelayInput) -> EnsembleOutput:
        """Executes all child operators sequentially and collects results.

        This implementation creates a natural data flow graph where
        each operation is independent but executed in sequence,
        making it an ideal candidate for parallel optimization.

        Args:
            inputs: Task context information containing task identifier

        Returns:
            Aggregated results from all child operations
        """
        results = []
        for i, op in enumerate(self.operators):
            # Creating a unique task ID for each child operation
            task_input = DelayInput(task_id=f"{inputs.task_id}-{i+1}")
            output = op(inputs=task_input)
            results.append(output.result)

        return EnsembleOutput(results=results, task_id=inputs.task_id)


@jit
class ParallelEnsembleOperator(Operator[DelayInput, EnsembleOutput]):
    """JIT-optimized operator explicitly using parallel execution.

    Demonstrates how operators can internally leverage concurrency
    while still benefiting from JIT tracing and optimization.
    This implementation explicitly uses thread-based parallelism
    to execute child operators concurrently.

    Attributes:
        specification: The operator's input/output contract
        operators: Collection of child delay operators to execute
        num_ops: Maximum number of concurrent operations
    """

    specification: ClassVar[Specification] = EnsembleSpecification()

    # Field declarations
    operators: List[DelayOperator]
    num_ops: int

    def __init__(self, *, num_ops: int = 3, delay: float = 0.1) -> None:
        """Initializes the ensemble with a collection of delay operators.

        Args:
            num_ops: Number of child operators to create and maximum concurrency
            delay: Delay duration for each child operator in seconds
        """
        self.operators = [
            DelayOperator(delay_seconds=delay, op_id=f"Op-{i+1}")
            for i in range(num_ops)
        ]
        self.num_ops = num_ops

    def forward(self, *, inputs: DelayInput) -> EnsembleOutput:
        """Executes all child operators concurrently and collects results.

        Uses explicit threading to achieve parallelism, demonstrating
        how operators can internally optimize execution while still
        participating in the JIT tracing ecosystem.

        Args:
            inputs: Task context information containing task identifier

        Returns:
            Aggregated results from all child operations
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.num_ops) as executor:
            # Submitting all tasks to the executor
            futures = []
            for i, op in enumerate(self.operators):
                task_input = DelayInput(task_id=f"{inputs.task_id}-{i+1}")
                futures.append(executor.submit(op, inputs=task_input))

            # Collecting results as they complete
            for future in as_completed(futures):
                output = future.result()
                results.append(output.result)

        return EnsembleOutput(results=results, task_id=inputs.task_id)


def run_benchmark(
    *,
    name: str,
    operator: Operator,
    num_runs: int = 2,
    task_prefix: str = "run",
) -> Tuple[List[float], List[str]]:
    """Executes a benchmark of the provided operator.

    Runs the given operator multiple times and collects performance metrics
    and results. The first run typically includes JIT tracing overhead,
    while subsequent runs benefit from the cached execution plan.

    Args:
        name: Descriptive name for the benchmark
        operator: Operator instance to benchmark
        num_runs: Number of consecutive executions to measure
        task_prefix: Prefix for task identifiers

    Returns:
        Tuple containing (list of execution times, list of result strings)
    """
    times = []
    all_results = []

    logger.info("\n%s benchmark:", name)
    for i in range(num_runs):
        run_label = f"Run {i+1}" if i > 0 else "First run (with tracing)"
        logger.info("  %s...", run_label)

        start = time.time()
        result = operator(inputs=DelayInput(task_id=f"{task_prefix}-{i+1}"))
        elapsed = time.time() - start

        times.append(elapsed)
        all_results.extend(result.results)

        logger.info(
            "    Completed %d operations in %.4fs", len(result.results), elapsed
        )

    return times, all_results


def analyze_performance(
    *,
    name: str,
    times: List[float],
    baseline_time: Optional[float] = None,
    theoretical_time: Optional[float] = None,
) -> None:
    """Analyzes and reports performance metrics.

    Calculates and logs various performance metrics comparing the actual
    execution times against theoretical optimums and baselines.

    Args:
        name: Name of the execution strategy being analyzed
        times: List of execution times to analyze (first time typically includes tracing)
        baseline_time: Optional baseline time for comparison
        theoretical_time: Optional theoretical optimal time
    """
    if not times:
        return

    first_time = times[0]

    if len(times) > 1:
        subsequent_avg = sum(times[1:]) / len(times[1:])
        speedup = first_time / subsequent_avg if subsequent_avg > 0 else 0

        logger.info("\n%s Analysis:", name)
        logger.info("  First run:            %.4fs", first_time)
        logger.info("  Subsequent runs avg:  %.4fs", subsequent_avg)

        if speedup > 1.05:  # Only reporting meaningful speedups
            improvement = (first_time - subsequent_avg) / first_time
            logger.info(
                "  JIT caching benefit:  %.1f%% faster after tracing", improvement * 100
            )

    if baseline_time is not None and baseline_time > 0:
        best_time = min(times)
        speedup = baseline_time / best_time if best_time > 0 else 0

        if speedup > 1.05:  # Only reporting meaningful speedups
            improvement = (baseline_time - best_time) / baseline_time
            logger.info(
                "  Speedup vs baseline: %.1f%% faster (%.1fx)",
                improvement * 100,
                speedup,
            )

    if theoretical_time is not None and theoretical_time > 0:
        best_time = min(times[1:]) if len(times) > 1 else first_time
        accuracy = best_time / theoretical_time

        logger.info("  Theoretical optimum: %.4fs", theoretical_time)
        logger.info(
            "  Achieved efficiency: %.1f%% of theoretical optimum", (1 / accuracy) * 100
        )


def demo_sequential_vs_jit(*, num_ops: int, delay: float) -> None:
    """Demonstrates basic JIT functionality without parallelism.

    Compares sequential execution time to JIT execution,
    showing the caching and optimization benefits of JIT
    even without explicit parallelism. Shows how the first run
    with tracing differs from subsequent cached runs.

    Args:
        num_ops: Number of operators to use in the ensemble
        delay: Delay duration for each operator in seconds
    """
    logger.info("\n%s", "=" * 80)
    logger.info(
        "DEMO 1: Sequential vs JIT - %d operators, %.3fs delay each", num_ops, delay
    )
    logger.info("%s", "=" * 80)
    logger.info(
        "Demonstrating JIT caching and optimization benefits without parallelism."
    )

    # Creating the JIT operator
    jit_op = JITEnsembleOperator(num_ops=num_ops, delay=delay)

    # Running benchmark
    times, _ = run_benchmark(
        name="Sequential JIT Ensemble",
        operator=jit_op,
        num_runs=3,
        task_prefix="sequential",
    )

    # Calculating theoretical sequential time
    theoretical_time = num_ops * delay

    # Analyzing performance
    analyze_performance(
        name="Sequential JIT",
        times=times,
        theoretical_time=theoretical_time,
    )


def demo_explicit_parallel_execution(*, num_ops: int, delay: float) -> None:
    """Demonstrates JIT with explicit graph-based parallel execution.

    Shows how the XCS engine can transform a sequential execution
    graph into a parallel execution plan automatically. Compares
    the performance of sequential and parallel schedulers on
    the same computation graph.

    Args:
        num_ops: Number of operators to use in the ensemble
        delay: Delay duration for each operator in seconds
    """
    logger.info("\n%s", "=" * 80)
    logger.info(
        "DEMO 2: Sequential vs Parallel Execution - %d operators, %.3fs delay each",
        num_ops,
        delay,
    )
    logger.info("%s", "=" * 80)
    logger.info(
        "Demonstrating how JIT + parallel scheduling transforms execution graphs."
    )

    # Creating the ensemble operator
    ensemble = JITEnsembleOperator(num_ops=num_ops, delay=delay)

    # First tracing the execution to build a graph
    with TracerContext() as tracer:
        # Executing the operator to capture the trace
        _ = ensemble(inputs=DelayInput(task_id="trace-run"))

    logger.info("\nCaptured %d trace records", len(tracer.records))

    # Building a graph directly from operators
    logger.info("\nBuilding execution graph...")
    graph = XCSGraph()

    # Adding each operator as a separate node
    for i, op in enumerate(ensemble.operators):
        node_id = f"node_{i}"
        graph.add_node(
            operator=lambda inputs, op=op: op(
                inputs=DelayInput(task_id=inputs["task_id"])
            ),
            node_id=node_id,
        )

    # No need to explicitly compile the graph with the unified engine
    # execute_graph handles graph compilation internally

    # Executing with different schedulers
    logger.info("\nExecuting with different schedulers:")

    # Sequential execution
    logger.info("  Sequential scheduler...")
    start = time.time()
    _ = execute_graph(
        graph=graph, 
        inputs={"task_id": "sequential"}, 
        options=ExecutionOptions(scheduler="wave")
    )
    seq_time = time.time() - start
    logger.info("    Completed in %.4fs", seq_time)

    # Parallel execution
    logger.info("  Parallel scheduler...")
    start = time.time()
    _ = execute_graph(
        graph=graph, 
        inputs={"task_id": "parallel"}, 
        options=ExecutionOptions(scheduler="parallel", max_workers=num_ops)
    )
    par_time = time.time() - start
    logger.info("    Completed in %.4fs", par_time)

    # Analysis
    theoretical_sequential = num_ops * delay
    theoretical_parallel = delay  # Theoretical: all operations run in parallel

    logger.info("\nExecution strategy comparison:")
    logger.info("  Sequential execution:    %.4fs", seq_time)
    logger.info("  Parallel execution:      %.4fs", par_time)
    logger.info("  Theoretical sequential:  %.4fs", theoretical_sequential)
    logger.info("  Theoretical parallel:    %.4fs", theoretical_parallel)

    if par_time < seq_time:
        speedup = seq_time / par_time
        improvement = (seq_time - par_time) / seq_time
        logger.info(
            "\n✅ Parallel scheduler achieved %.1f%% improvement (%.1fx)",
            improvement * 100,
            speedup,
        )

    # Efficiency analysis
    seq_efficiency = theoretical_sequential / seq_time if seq_time > 0 else 0
    par_efficiency = theoretical_parallel / par_time if par_time > 0 else 0

    logger.info("\nEfficiency analysis:")
    logger.info(
        "  Sequential scheduler: %.1f%% of theoretical optimum", seq_efficiency * 100
    )
    logger.info(
        "  Parallel scheduler:   %.1f%% of theoretical optimum", par_efficiency * 100
    )


def demo_execution_strategies(*, num_ops: int, delay: float) -> None:
    """Demonstrates different operator execution strategies.

    Compares sequential, JIT-optimized, and explicitly parallel
    implementations to highlight performance characteristics.
    Also showcases a context-aware operator that can adapt its
    execution strategy based on runtime configuration.

    Args:
        num_ops: Number of operators to use in the ensemble
        delay: Delay duration for each operator in seconds
    """
    logger.info("\n%s", "=" * 80)
    logger.info(
        "DEMO 3: Execution Strategy Comparison - %d operators, %.3fs delay each",
        num_ops,
        delay,
    )
    logger.info("%s", "=" * 80)
    logger.info(
        "Comparing different execution strategies with the same logical operations."
    )

    # Creating operators with different execution strategies
    sequential_op = JITEnsembleOperator(num_ops=num_ops, delay=delay)
    parallel_op = ParallelEnsembleOperator(num_ops=num_ops, delay=delay)

    # Creating a context-dependent operator that adapts based on execution options
    @jit
    class ContextAwareOperator(Operator[DelayInput, EnsembleOutput]):
        """Operator that adapts execution strategy based on context.

        Demonstrates how operators can dynamically select their execution
        strategy based on runtime context, allowing the same logical
        operation to be optimized for different execution environments.

        Attributes:
            specification: The operator's input/output contract
            operators: Collection of child delay operators to execute
            num_ops: Maximum number of concurrent operations
        """

        specification: ClassVar[Specification] = EnsembleSpecification()

        # Field declarations
        operators: List[DelayOperator]
        num_ops: int

        def __init__(self, *, num_ops: int = 3, delay: float = 0.1) -> None:
            """Initializes with configurable operators.

            Args:
                num_ops: Number of child operators to create and maximum concurrency
                delay: Delay duration for each child operator in seconds
            """
            self.operators = [
                DelayOperator(delay_seconds=delay, op_id=f"Op-{i+1}")
                for i in range(num_ops)
            ]
            self.num_ops = num_ops

        def forward(self, *, inputs: Union[DelayInput, Dict[str, Any]]) -> EnsembleOutput:
            """Executes with strategy based on context.

            Checks the execution context for configuration parameters
            that indicate whether to use parallel execution, and
            dynamically selects the appropriate strategy.

            Args:
                inputs: Task context information containing task identifier,
                       either as DelayInput object or dictionary

            Returns:
                Aggregated results from all child operations
            """
            results = []
            
            # Extract task_id from inputs, handling both object and dict formats
            if isinstance(inputs, dict) and "task_id" in inputs:
                task_id = inputs["task_id"]
            elif hasattr(inputs, "task_id"):
                task_id = inputs.task_id
            else:
                task_id = "unknown-task"

            # Checking execution context for parallelism hints
            current_options = get_execution_options()
            use_parallel = current_options.use_parallel

            if use_parallel:
                # Parallel execution strategy using threading
                with ThreadPoolExecutor(max_workers=self.num_ops) as executor:
                    futures = []
                    for i, op in enumerate(self.operators):
                        task_input = DelayInput(task_id=f"{task_id}-{i+1}")
                        futures.append(executor.submit(op, inputs=task_input))

                    for future in as_completed(futures):
                        output = future.result()
                        results.append(output.result)
            else:
                # Sequential execution strategy
                for i, op in enumerate(self.operators):
                    task_input = DelayInput(task_id=f"{task_id}-{i+1}")
                    output = op(inputs=task_input)
                    results.append(output.result)

            return EnsembleOutput(results=results, task_id=task_id)

    # Creating the context-aware operator
    adaptive_op = ContextAwareOperator(num_ops=num_ops, delay=delay)

    # Benchmarking sequential operator
    seq_times, _ = run_benchmark(
        name="Sequential Operator",
        operator=sequential_op,
        num_runs=2,
        task_prefix="seq",
    )

    # Benchmarking parallel operator
    par_times, _ = run_benchmark(
        name="Parallel Operator",
        operator=parallel_op,
        num_runs=2,
        task_prefix="par",
    )

    # Benchmarking adaptive operator with sequential context
    logger.info("\nAdaptive Operator (Sequential Context):")
    start = time.time()
    # Use execution options context manager for this execution
    with execution_options(use_parallel=False):
        _ = adaptive_op(inputs=DelayInput(task_id="adaptive-seq"))
    adaptive_seq_time = time.time() - start
    logger.info("  Completed in %.4fs", adaptive_seq_time)

    # Benchmarking adaptive operator with parallel context
    logger.info("\nAdaptive Operator (Parallel Context):")
    start = time.time()
    # Use execution options context manager for this execution
    with execution_options(use_parallel=True):
        _ = adaptive_op(inputs=DelayInput(task_id="adaptive-par"))
    adaptive_par_time = time.time() - start
    logger.info("  Completed in %.4fs", adaptive_par_time)

    # Performance analysis
    seq_best = min(seq_times)
    par_best = min(par_times)
    theoretical = delay  # Optimal parallel execution time

    logger.info("\nExecution Strategy Comparison:")
    logger.info("  Sequential best:       %.4fs", seq_best)
    logger.info("  Parallel best:         %.4fs", par_best)
    logger.info("  Adaptive (sequential): %.4fs", adaptive_seq_time)
    logger.info("  Adaptive (parallel):   %.4fs", adaptive_par_time)
    logger.info("  Theoretical optimum:   %.4fs", theoretical)

    # Calculating and reporting speedups
    if seq_best > 0:
        best_overall = min(par_best, adaptive_par_time)
        overall_speedup = seq_best / best_overall if best_overall > 0 else 0

        if overall_speedup > 1.05:
            logger.info(
                "\n✅ Parallelism achieved %.1fx speedup vs sequential", overall_speedup
            )

    # Context adaptability analysis
    if adaptive_seq_time > 0 and adaptive_par_time > 0:
        context_adaptability = adaptive_seq_time / adaptive_par_time

        if context_adaptability > 1.05:
            logger.info(
                "\n✅ Context-aware execution demonstrated %.1fx speedup through adaptability",
                context_adaptability,
            )


def main() -> None:
    """Executes the JIT optimization demonstration suite.

    Runs a series of demonstrations showcasing different aspects
    of JIT optimization and parallel execution strategies. Each
    demonstration highlights specific capabilities of the XCS
    engine and JIT tracing system.
    """
    logger.info("Just-In-Time Optimization Demonstration")
    logger.info("=======================================")
    logger.info("\nThis demonstration shows how JIT tracing and compilation")
    logger.info("enables significant performance optimizations through:")
    logger.info(" - Execution graph analysis and caching")
    logger.info(" - Automatic parallelization of independent operations")
    logger.info(" - Context-aware execution strategy selection")

    # Configuration parameters
    num_ops = 10
    delay = 0.1

    # Demo 1: Basic JIT functionality
    demo_sequential_vs_jit(num_ops=num_ops, delay=delay)

    # Demo 2: Manual parallel execution
    demo_explicit_parallel_execution(num_ops=num_ops, delay=delay)

    # Demo 3: Different execution strategies
    demo_execution_strategies(num_ops=num_ops, delay=delay)

    logger.info("\n%s", "=" * 80)
    logger.info("KEY INSIGHTS:")
    logger.info(
        "1. JIT tracing converts imperative code into optimizable computation graphs"
    )
    logger.info(
        "2. Independent operations are automatically identified for parallelization"
    )
    logger.info("3. Execution strategies can be selected based on runtime context")
    logger.info(
        "4. First-run tracing overhead is amortized through cached execution plans"
    )
    logger.info(
        "5. Context-aware operators can dynamically adapt to execution environments"
    )

    logger.info("\nPRACTICAL APPLICATIONS:")
    logger.info(
        "- Complex ML inference pipelines with many independent processing steps"
    )
    logger.info(
        "- Data transformation workflows with embarrassingly parallel operations"
    )
    logger.info("- Ensemble methods combining multiple model predictions")
    logger.info(
        "- Systems that need to adapt to different hardware capabilities at runtime"
    )


if __name__ == "__main__":
    main()
