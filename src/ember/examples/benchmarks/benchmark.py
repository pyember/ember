"""
Simple benchmark script for enhanced JIT parallelization.
"""

import time
import concurrent.futures
from typing import Any, Dict, List


class SimpleOperator:
    """Simple operator with configurable delay."""
    
    def __init__(self, name: str, delay: float = 0.1):
        self.name = name
        self.delay = delay
    
    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        time.sleep(self.delay)
        value = inputs.get("value", "")
        return {"result": f"{self.name}:{value}", "source": self.name}


class SequentialEnsemble:
    """Sequential ensemble using standard loop pattern."""
    
    def __init__(self, num_operators: int = 3, delay: float = 0.1):
        self.operators = [
            SimpleOperator(f"op{i}", delay)
            for i in range(num_operators)
        ]
    
    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = []
        
        for op in self.operators:
            result = op(inputs=inputs)
            results.append(result["result"])
        
        return {"results": results}


class ParallelEnsemble:
    """Parallel ensemble using threading."""
    
    def __init__(self, num_operators: int = 3, delay: float = 0.1):
        self.operators = [
            SimpleOperator(f"op{i}", delay)
            for i in range(num_operators)
        ]
    
    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = [None] * len(self.operators)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            
            for i, op in enumerate(self.operators):
                future = executor.submit(op, inputs=inputs)
                futures[future] = i
            
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    result = future.result()
                    results[i] = result["result"]
                except Exception as e:
                    results[i] = f"Error: {str(e)}"
        
        return {"results": results}


def run_benchmark(num_operators: int = 5, num_runs: int = 3):
    """Run benchmark comparing sequential vs parallel execution."""
    print(f"Running benchmark with {num_operators} operators, {num_runs} runs...")
    
    seq_ensemble = SequentialEnsemble(num_operators=num_operators, delay=0.05)
    par_ensemble = ParallelEnsemble(num_operators=num_operators, delay=0.05)
    
    inputs = {"value": "test"}
    
    seq_times = []
    par_times = []
    
    for i in range(num_runs):
        print(f"\nRun {i+1}/{num_runs}:")
        
        # Sequential execution
        print("Sequential execution...")
        start = time.time()
        seq_result = seq_ensemble(inputs=inputs)
        seq_time = time.time() - start
        seq_times.append(seq_time)
        print(f"Took {seq_time:.4f}s")
        
        # Parallel execution
        print("Parallel execution...")
        start = time.time()
        par_result = par_ensemble(inputs=inputs)
        par_time = time.time() - start
        par_times.append(par_time)
        print(f"Took {par_time:.4f}s")
        
        # Calculate speedup
        speedup = seq_time / par_time
        print(f"Speedup: {speedup:.2f}x")
    
    # Calculate average times
    avg_seq_time = sum(seq_times) / len(seq_times)
    avg_par_time = sum(par_times) / len(par_times)
    avg_speedup = avg_seq_time / avg_par_time
    
    print("\nSummary:")
    print(f"Average sequential time: {avg_seq_time:.4f}s")
    print(f"Average parallel time: {avg_par_time:.4f}s")
    print(f"Average speedup: {avg_speedup:.2f}x")
    
    # Theoretical maximum speedup
    print(f"Theoretical maximum speedup: {num_operators:.2f}x")
    print(f"Efficiency: {(avg_speedup / num_operators) * 100:.1f}%")


def run_scaling_test():
    """Run tests with different numbers of operators to see how speedup scales."""
    print("=== SCALING TEST ===")
    
    operator_counts = [2, 5, 10, 20]
    results = []
    
    for count in operator_counts:
        print(f"\n==== Testing with {count} operators ====")
        
        seq_ensemble = SequentialEnsemble(num_operators=count, delay=0.05)
        par_ensemble = ParallelEnsemble(num_operators=count, delay=0.05)
        
        inputs = {"value": "test"}
        
        # Run sequential
        start = time.time()
        seq_ensemble(inputs=inputs)
        seq_time = time.time() - start
        
        # Run parallel
        start = time.time()
        par_ensemble(inputs=inputs)
        par_time = time.time() - start
        
        # Calculate speedup
        speedup = seq_time / par_time
        efficiency = (speedup / count) * 100
        
        print(f"Sequential: {seq_time:.4f}s")
        print(f"Parallel: {par_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x (Efficiency: {efficiency:.1f}%)")
        
        results.append((count, seq_time, par_time, speedup, efficiency))
    
    # Print summary table
    print("\n=== RESULTS SUMMARY ===")
    print("Operators | Sequential | Parallel | Speedup | Efficiency")
    print("----------|------------|----------|---------|----------")
    
    for count, seq_time, par_time, speedup, efficiency in results:
        print(f"{count:9d} | {seq_time:.4f}s    | {par_time:.4f}s  | {speedup:.2f}x   | {efficiency:.1f}%")


if __name__ == "__main__":
    # Run the standard benchmark
    run_benchmark(num_operators=5, num_runs=3)
    
    print("\n" + "=" * 50 + "\n")
    
    # Run the scaling test
    run_scaling_test()