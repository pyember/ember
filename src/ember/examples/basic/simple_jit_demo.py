"""Enhanced JIT Demo with Parallelism

This module demonstrates how JIT delivers substantial performance improvements 
through graph compilation and parallel execution. It uses the XCS engine's native
parallel scheduler to execute operators concurrently.

Key insight: JIT traces operator calls into a computation graph that can be executed
with different schedulers. When using the parallel scheduler, independent operations
can run concurrently, significantly improving performance for suitable workloads.

To run:
    poetry run python src/ember/examples/simple_jit_demo.py
"""

import time
from typing import Any, Dict, List, ClassVar, Type, Optional

# Import the JIT decorator and execution components
from ember.api.xcs import jit
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.operator.base.operator_base import Specification
from ember.core.types.ember_model import EmberModel, Field
from ember.xcs.engine.execution_options import execution_options
from ember.xcs.tracer.xcs_tracing import TracerContext
from ember.xcs.engine.xcs_engine import (
    compile_graph,
    TopologicalSchedulerWithParallelDispatch,
)
from ember.xcs.engine.xcs_noop_scheduler import XCSNoOpScheduler
from ember.xcs.graph.xcs_graph import XCSGraph


# Define models for our operators
class DelayInput(EmberModel):
    """Input for the delay operator."""

    task_id: str = Field(description="Identifier for the task")


class DelayOutput(EmberModel):
    """Output from the delay operator."""

    result: str = Field(description="The result of processing")
    task_id: str = Field(description="Identifier for the task")


class EnsembleOutput(EmberModel):
    """Output from the ensemble operator."""

    results: List[str] = Field(description="List of results from child operations")
    task_id: str = Field(description="Identifier for the task")


# Define specifications
class DelaySpecification(Specification):
    """Specification for delay operator."""

    input_model: Type[EmberModel] = DelayInput
    structured_output: Type[EmberModel] = DelayOutput


class EnsembleSpecification(Specification):
    """Specification for ensemble operator."""

    input_model: Type[EmberModel] = DelayInput
    structured_output: Type[EmberModel] = EnsembleOutput


# Define a simple operator that introduces a delay
class DelayOperator(Operator):
    """Operator that sleeps for a specified time and returns a result."""

    specification: ClassVar[Specification] = DelaySpecification()

    def __init__(self, *, delay_seconds: float, op_id: str) -> None:
        """Initialize with delay time in seconds and an identifier."""
        self.delay_seconds = delay_seconds
        self.op_id = op_id

    def forward(self, *, inputs: DelayInput) -> DelayOutput:
        """Execute with a delay and return result."""
        time.sleep(self.delay_seconds)
        return DelayOutput(
            result=f"Operator {self.op_id} completed task {inputs.task_id}",
            task_id=inputs.task_id,
        )


# Apply JIT to a composite operator
@jit
class JITEnsembleOperator(Operator):
    """A JIT-decorated operator that composes multiple DelayOperators.

    When executed, this operator creates a trace that can be converted
    to an XCS graph and executed with different schedulers.
    """

    specification: ClassVar[Specification] = EnsembleSpecification()

    def __init__(self, *, num_ops: int = 3, delay: float = 0.1) -> None:
        """Initialize with multiple delay operators."""
        self.operators = [
            DelayOperator(delay_seconds=delay, op_id=f"Op-{i+1}")
            for i in range(num_ops)
        ]

    def forward(self, *, inputs: DelayInput) -> EnsembleOutput:
        """Execute all internal operators and return combined results."""
        # This simple pattern of independent operator calls
        # is ideal for parallel execution
        results = []
        for i, op in enumerate(self.operators):
            # Each operator gets its own task_id
            task_input = DelayInput(task_id=f"{inputs.task_id}-{i+1}")
            output = op(inputs=task_input)
            results.append(output.result)

        return EnsembleOutput(results=results, task_id=inputs.task_id)


def demo_sequential_vs_jit(*, num_ops: int, delay: float) -> None:
    """Compare sequential execution to JIT execution."""
    print(f"\n{'=' * 80}")
    print(f"DEMO 1: Sequential vs JIT - {num_ops} operators, {delay:.3f}s delay each")
    print(f"{'=' * 80}")
    print("This demo shows the basic JIT functionality without parallelism.")

    # Create the JIT operator
    jit_op = JITEnsembleOperator(num_ops=num_ops, delay=delay)

    # First run with tracing
    print("\nFirst execution (JIT will trace and compile):")
    start = time.time()
    result1 = jit_op(inputs=DelayInput(task_id="first-run"))
    first_time = time.time() - start
    print(f"  Results: {len(result1.results)} operations completed")
    print(f"  First execution time: {first_time:.4f}s")

    # Second run with cached plan
    print("\nSecond execution (JIT should use cached plan):")
    start = time.time()
    result2 = jit_op(inputs=DelayInput(task_id="second-run"))
    second_time = time.time() - start
    print(f"  Results: {len(result2.results)} operations completed")
    print(f"  Second execution time: {second_time:.4f}s")

    # Expected sequential time
    expected_time = num_ops * delay
    print(f"\nExpected sequential time: {expected_time:.4f}s")
    print(f"Actual times: First={first_time:.4f}s, Second={second_time:.4f}s")

    # Analysis
    if second_time < first_time:
        improvement = (first_time - second_time) / first_time
        print(f"\n✅ Second run was {improvement:.1%} faster than first run")
        print("   This shows the benefit of JIT's cached execution plan")
    else:
        print("\nℹ️ No significant speedup between runs")
        print("   This is expected for simple sequential workloads")


def demo_parallel_execution(*, num_ops: int, delay: float) -> None:
    """Demonstrate JIT with explicit parallel execution through the XCS engine."""
    print(f"\n{'=' * 80}")
    print(
        f"DEMO 2: Sequential vs Parallel Execution - {num_ops} operators, {delay:.3f}s delay each"
    )
    print(f"{'=' * 80}")
    print(
        "This demo shows how JIT + parallel scheduling drastically improves performance."
    )

    # Create the ensemble
    ensemble = JITEnsembleOperator(num_ops=num_ops, delay=delay)

    # Trace the execution to build a graph
    with TracerContext() as tracer:
        # First, execute the operator to capture the trace
        _ = ensemble(inputs=DelayInput(task_id="trace-run"))

    print(f"\nCaptured {len(tracer.records)} trace records")

    # Build a graph directly from operators
    print("\nBuilding execution graph...")
    graph = XCSGraph()

    # Add each operator as a separate node
    for i, op in enumerate(ensemble.operators):
        node_id = f"node_{i}"
        graph.add_node(
            operator=lambda inputs, op=op: op(
                inputs=DelayInput(task_id=inputs["task_id"])
            ),
            node_id=node_id,
        )

    # Compile the graph
    plan = compile_graph(graph=graph)

    # Execute sequentially
    print("\nExecuting sequentially...")
    seq_scheduler = XCSNoOpScheduler()
    start = time.time()
    _ = seq_scheduler.run_plan(
        plan=plan, global_input={"task_id": "sequential"}, graph=graph
    )
    seq_time = time.time() - start
    print(f"Sequential execution time: {seq_time:.4f}s")

    # Execute in parallel
    print("\nExecuting in parallel...")
    par_scheduler = TopologicalSchedulerWithParallelDispatch(max_workers=num_ops)
    start = time.time()
    _ = par_scheduler.run_plan(
        plan=plan, global_input={"task_id": "parallel"}, graph=graph
    )
    par_time = time.time() - start
    print(f"Parallel execution time: {par_time:.4f}s")

    # Analysis
    expected_sequential = num_ops * delay
    expected_parallel = delay  # Theoretical: all operations run in parallel

    print(f"\nTheoretical sequential time: {expected_sequential:.4f}s")
    print(f"Theoretical parallel time:   {expected_parallel:.4f}s")
    print(f"Theoretical speedup:         {expected_sequential/expected_parallel:.1f}x")

    if par_time < seq_time:
        speedup = seq_time / par_time
        improvement = (seq_time - par_time) / seq_time
        print(f"\n✅ Parallel execution was {improvement:.1%} faster!")
        print(f"   Achieved {speedup:.1f}x speedup vs. sequential execution")
    else:
        print("\n⚠️ Parallel execution not faster for this workload")

    # Compare to expected
    seq_accuracy = abs(seq_time - expected_sequential) / expected_sequential
    par_accuracy = abs(par_time - expected_parallel) / expected_parallel

    print(f"\nSequential execution was within {seq_accuracy:.1%} of theoretical time")
    print(f"Parallel execution was within {par_accuracy:.1%} of theoretical time")


def demo_jit_with_concurrent_execution(*, num_ops: int, delay: float) -> None:
    """Demonstrate JIT with a concurrent operator implementation."""
    print(f"\n{'=' * 80}")
    print(
        f"DEMO 3: JIT with Concurrent Execution - {num_ops} operators, {delay:.3f}s delay each"
    )
    print(f"{'=' * 80}")
    print("This demo shows a more practical approach to concurrent execution with JIT.")

    # Define a concurrent version of the JIT operator
    @jit
    class ConcurrentJITOperator(Operator):
        """A JIT-decorated operator that runs tasks concurrently."""

        specification: ClassVar[Specification] = EnsembleSpecification()

        def __init__(self, *, num_ops: int = 3, delay: float = 0.1) -> None:
            """Initialize with multiple delay operators."""
            self.operators = [
                DelayOperator(delay_seconds=delay, op_id=f"Op-{i+1}")
                for i in range(num_ops)
            ]
            self.num_ops = num_ops

        def forward(self, *, inputs: DelayInput) -> EnsembleOutput:
            """Execute all internal operators concurrently and return combined results."""
            results = []

            # Use concurrent execution with a thread pool
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_ops
            ) as executor:
                # Submit all tasks to the executor
                futures = []
                for i, op in enumerate(self.operators):
                    task_input = DelayInput(task_id=f"{inputs.task_id}-{i+1}")
                    futures.append(executor.submit(op, inputs=task_input))

                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    output = future.result()
                    results.append(output.result)

            return EnsembleOutput(results=results, task_id=inputs.task_id)

    # Create operators
    sequential_op = JITEnsembleOperator(num_ops=num_ops, delay=delay)
    concurrent_op = ConcurrentJITOperator(num_ops=num_ops, delay=delay)

    # Sequential execution
    print("\nSequential execution:")
    start = time.time()
    _ = sequential_op(inputs=DelayInput(task_id="sequential"))
    seq_time = time.time() - start
    print(f"Sequential execution time: {seq_time:.4f}s")

    # First concurrent execution (with tracing)
    print("\nFirst concurrent execution (with tracing):")
    start = time.time()
    _ = concurrent_op(inputs=DelayInput(task_id="concurrent-first"))
    conc_time1 = time.time() - start
    print(f"First concurrent execution time: {conc_time1:.4f}s")

    # Second concurrent execution (cached plan)
    print("\nSecond concurrent execution (cached plan):")
    start = time.time()
    _ = concurrent_op(inputs=DelayInput(task_id="concurrent-second"))
    conc_time2 = time.time() - start
    print(f"Second concurrent execution time: {conc_time2:.4f}s")

    # Analysis
    expected_sequential = num_ops * delay
    expected_concurrent = delay

    print(f"\nTheoretical sequential time: {expected_sequential:.4f}s")
    print(f"Theoretical concurrent time:   {expected_concurrent:.4f}s")

    if conc_time2 < seq_time:
        speedup = seq_time / conc_time2
        improvement = (seq_time - conc_time2) / seq_time
        print(f"\n✅ Concurrent execution was {improvement:.1%} faster!")
        print(f"   Achieved {speedup:.1f}x speedup vs. sequential execution")
    else:
        print("\n⚠️ Concurrent execution not faster for this workload")

    if conc_time2 < conc_time1:
        improvement = (conc_time1 - conc_time2) / conc_time1
        print(f"\n✅ Second concurrent run was {improvement:.1%} faster than first run")
        print("   This shows the benefit of JIT's cached execution plan")


def main() -> None:
    """Run the JIT demonstration with various configurations."""
    print("Enhanced JIT Demo with Parallelism")
    print("===================================")
    print("\nThis demo shows how JIT compilation and parallel execution")
    print("combine to deliver substantial performance improvements.")

    # Demo 1: Basic JIT functionality (sequential)
    demo_sequential_vs_jit(num_ops=10, delay=0.1)

    # Demo 2: Manual parallel execution with XCS engine
    demo_parallel_execution(num_ops=10, delay=0.1)

    # Demo 3: JIT with concurrent execution
    demo_jit_with_concurrent_execution(num_ops=10, delay=0.1)

    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("1. JIT alone provides minimal benefits for sequential execution")
    print("2. JIT + parallel scheduling delivers dramatic performance improvements")
    print("3. The XCS engine enables different execution strategies for the same code")
    print("4. Independent operations are ideal candidates for parallelization")
    print("5. First execution has tracing overhead, subsequent executions are faster")

    print("\nWHEN TO USE JIT:")
    print("- Complex computation graphs with many operations")
    print("- Independent operations that can run in parallel")
    print("- Operations that will be executed multiple times with similar patterns")
    print("- When optimizing throughput for scale is important")


if __name__ == "__main__":
    main()
