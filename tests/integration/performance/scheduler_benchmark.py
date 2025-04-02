"""Scheduler benchmark script.

This script benchmarks the schedulers with a simple synthetic workload
designed to measure the performance benefits of parallel execution.
"""

import time
import statistics
from typing import Any, ClassVar, Dict, List, Type, cast

from ember.api.operators import Operator
from ember.core.registry.operator.base.operator_base import Specification
from ember.core.registry.specification.specification import Specification as CoreSpecification
from ember.core.types.ember_model import EmberModel
from ember.xcs.engine.execution_options import execution_options


# Define simple data models
class BenchmarkInput(EmberModel):
    """Input for benchmark operator."""
    delay: float = 0.1
    count: int = 5


class DelayOutput(EmberModel):
    """Output from delay operator."""
    result: float


class BenchmarkOutput(EmberModel):
    """Output from benchmark operator."""
    results: List[DelayOutput]


class BenchmarkSpec(CoreSpecification):
    """Specification for benchmark operator."""
    input_model: Type[BenchmarkInput] = BenchmarkInput
    structured_output: Type[BenchmarkOutput] = BenchmarkOutput


class DelayOperator(Operator[BenchmarkInput, DelayOutput]):
    """Simple operator that just waits for a specified delay."""
    
    specification: ClassVar[Specification] = CoreSpecification(
        input_model=BenchmarkInput,
        structured_output=DelayOutput,
    )
    
    def forward(self, *, inputs: BenchmarkInput) -> DelayOutput:
        """Wait for specified delay and return a timestamp."""
        time.sleep(inputs.delay)
        return DelayOutput(result=time.time())


class SequentialBenchmarkOperator(Operator[BenchmarkInput, BenchmarkOutput]):
    """Benchmark operator that executes sequentially.
    
    This operator creates multiple delay operators and calls them in a loop,
    which doesn't provide the scheduler any opportunity for parallelization.
    """
    
    specification: ClassVar[Specification] = BenchmarkSpec()
    
    def __init__(self) -> None:
        """Initialize with delay operators."""
        self.operators = [DelayOperator() for _ in range(10)]
    
    def forward(self, *, inputs: BenchmarkInput) -> BenchmarkOutput:
        """Run the benchmark sequentially."""
        results = []
        
        # Execute operators in a loop - scheduler can't parallelize this
        for i in range(inputs.count):
            result = self.operators[i % len(self.operators)](inputs=inputs)
            results.append(result)
            
        return BenchmarkOutput(results=results)


class ParallelizableBenchmarkOperator(Operator[BenchmarkInput, BenchmarkOutput]):
    """Benchmark operator that can be parallelized.
    
    This operator calls each delay operator separately, giving the scheduler
    a clear opportunity to parallelize the execution.
    """
    
    specification: ClassVar[Specification] = BenchmarkSpec()
    
    def __init__(self) -> None:
        """Initialize with named delay operators."""
        # Create operators with distinct names to aid the scheduler
        self.op1 = DelayOperator()
        self.op2 = DelayOperator()
        self.op3 = DelayOperator()
        self.op4 = DelayOperator()
        self.op5 = DelayOperator()
    
    def forward(self, *, inputs: BenchmarkInput) -> BenchmarkOutput:
        """Run the benchmark with parallelizable structure."""
        # These operations can be parallelized
        result1 = self.op1(inputs=inputs)
        result2 = self.op2(inputs=inputs)
        result3 = self.op3(inputs=inputs)
        result4 = self.op4(inputs=inputs)
        result5 = self.op5(inputs=inputs)
        
        results = [result1, result2, result3, result4, result5]
        
        return BenchmarkOutput(results=results)


def benchmark(iterations: int = 5) -> Dict[str, Dict[str, float]]:
    """Run benchmark comparing sequential and parallel execution.
    
    Args:
        iterations: Number of iterations to run for each configuration
        
    Returns:
        Dictionary of timing results
    """
    results = {
        "sequential_operator": {
            "sequential_scheduler": [],
            "parallel_scheduler": []
        },
        "parallelizable_operator": {
            "sequential_scheduler": [],
            "parallel_scheduler": []
        }
    }
    
    # Create benchmark input with small delay to keep test fast
    inputs = BenchmarkInput(delay=0.05, count=5)
    
    # Create operators
    sequential_op = SequentialBenchmarkOperator()
    parallelizable_op = ParallelizableBenchmarkOperator()
    
    # Test each configuration multiple times
    for _ in range(iterations):
        # Sequential operator with sequential scheduler
        start = time.time()
        with execution_options(scheduler="sequential"):
            sequential_op(inputs=inputs)
        results["sequential_operator"]["sequential_scheduler"].append(time.time() - start)
        
        # Sequential operator with parallel scheduler
        start = time.time()
        with execution_options(scheduler="parallel", max_workers=4):
            sequential_op(inputs=inputs)
        results["sequential_operator"]["parallel_scheduler"].append(time.time() - start)
        
        # Parallelizable operator with sequential scheduler
        start = time.time()
        with execution_options(scheduler="sequential"):
            parallelizable_op(inputs=inputs)
        results["parallelizable_operator"]["sequential_scheduler"].append(time.time() - start)
        
        # Parallelizable operator with parallel scheduler
        start = time.time()
        with execution_options(scheduler="parallel", max_workers=4):
            parallelizable_op(inputs=inputs)
        results["parallelizable_operator"]["parallel_scheduler"].append(time.time() - start)
    
    # Calculate statistics
    stats = {}
    for op_type, schedulers in results.items():
        stats[op_type] = {}
        for scheduler, times in schedulers.items():
            stats[op_type][scheduler] = {
                "mean": statistics.mean(times),
                "min": min(times),
                "max": max(times),
                "stddev": statistics.stdev(times) if len(times) > 1 else 0
            }
    
    return stats


def main() -> None:
    """Run the benchmark and print results."""
    print("Running scheduler benchmark...")
    stats = benchmark(iterations=5)
    
    # Print results
    print("\nResults (average execution time in seconds):")
    print("=" * 80)
    
    # Sequential operator results
    seq_seq = stats["sequential_operator"]["sequential_scheduler"]["mean"]
    seq_par = stats["sequential_operator"]["parallel_scheduler"]["mean"]
    seq_diff = seq_seq - seq_par
    seq_speedup = seq_seq / seq_par if seq_par > 0 else 0
    
    print(f"Sequential Operator:")
    print(f"  Sequential Scheduler: {seq_seq:.4f}s")
    print(f"  Parallel Scheduler:   {seq_par:.4f}s")
    print(f"  Difference:           {seq_diff:.4f}s ({seq_speedup:.2f}x speedup)")
    
    # Parallelizable operator results
    par_seq = stats["parallelizable_operator"]["sequential_scheduler"]["mean"]
    par_par = stats["parallelizable_operator"]["parallel_scheduler"]["mean"]
    par_diff = par_seq - par_par
    par_speedup = par_seq / par_par if par_par > 0 else 0
    
    print(f"\nParallelizable Operator:")
    print(f"  Sequential Scheduler: {par_seq:.4f}s")
    print(f"  Parallel Scheduler:   {par_par:.4f}s")
    print(f"  Difference:           {par_diff:.4f}s ({par_speedup:.2f}x speedup)")
    
    # Comparison of best configurations
    best_seq = min(seq_seq, seq_par)
    best_par = min(par_seq, par_par)
    overall_speedup = best_seq / best_par if best_par > 0 else 0
    
    print(f"\nOverall Improvement:")
    print(f"  Best Sequential Configuration: {best_seq:.4f}s")
    print(f"  Best Parallelizable Configuration: {best_par:.4f}s")
    print(f"  Speedup: {overall_speedup:.2f}x")
    

if __name__ == "__main__":
    main()