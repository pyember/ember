"""
Benchmark for IO-bound operations like LLM calls.
"""

import time
import random
import concurrent.futures
from typing import Any, Dict, List
import urllib.request
import json


class MockLLMOperator:
    """Simulates an LLM API call with network latency."""
    
    def __init__(self, name: str, base_latency: float = 0.2):
        self.name = name
        self.base_latency = base_latency
    
    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate network I/O with random latency."""
        # Simulate network latency with randomness like a real API
        latency = self.base_latency + random.uniform(0, 0.1)
        time.sleep(latency)
        
        # Simulate some CPU work
        result = f"Response from {self.name} to: {inputs.get('prompt', '')}"
        
        return {"response": result, "model": self.name}


class SequentialEnsemble:
    """Sequential ensemble using standard loop pattern."""
    
    def __init__(self, num_models: int = 3, base_latency: float = 0.2):
        self.operators = [
            MockLLMOperator(f"model{i}", base_latency)
            for i in range(num_models)
        ]
    
    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process sequentially with all operators."""
        results = []
        
        for op in self.operators:
            result = op(inputs=inputs)
            results.append(result["response"])
        
        return {"responses": results}


class ParallelEnsemble:
    """Parallel ensemble using threading."""
    
    def __init__(self, num_models: int = 3, base_latency: float = 0.2):
        self.operators = [
            MockLLMOperator(f"model{i}", base_latency)
            for i in range(num_models)
        ]
    
    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process in parallel with all operators."""
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
                    results[i] = result["response"]
                except Exception as e:
                    results[i] = f"Error: {str(e)}"
        
        return {"responses": results}


def run_io_benchmark(num_models: int = 5, num_runs: int = 3):
    """Run benchmark comparing sequential vs parallel execution for I/O bound operations."""
    print(f"Running I/O benchmark with {num_models} LLM calls, {num_runs} runs...")
    
    seq_ensemble = SequentialEnsemble(num_models=num_models, base_latency=0.2)
    par_ensemble = ParallelEnsemble(num_models=num_models, base_latency=0.2)
    
    # Test input like an LLM prompt
    inputs = {"prompt": "Explain quantum computing in simple terms"}
    
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
    print(f"Theoretical maximum speedup: {num_models:.2f}x")
    print(f"Efficiency: {(avg_speedup / num_models) * 100:.1f}%")


def run_io_scaling_test():
    """Run tests with different numbers of LLM calls to see how speedup scales."""
    print("=== I/O SCALING TEST ===")
    
    model_counts = [2, 5, 10, 20]
    results = []
    
    for count in model_counts:
        print(f"\n==== Testing with {count} LLM calls ====")
        
        seq_ensemble = SequentialEnsemble(num_models=count, base_latency=0.2)
        par_ensemble = ParallelEnsemble(num_models=count, base_latency=0.2)
        
        inputs = {"prompt": "Explain quantum computing in simple terms"}
        
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
    print("LLM Calls | Sequential | Parallel | Speedup | Efficiency")
    print("----------|------------|----------|---------|----------")
    
    for count, seq_time, par_time, speedup, efficiency in results:
        print(f"{count:9d} | {seq_time:.4f}s    | {par_time:.4f}s  | {speedup:.2f}x   | {efficiency:.1f}%")


if __name__ == "__main__":
    # Run the IO benchmark
    run_io_benchmark(num_models=5, num_runs=3)
    
    print("\n" + "=" * 50 + "\n")
    
    # Run the IO scaling test
    run_io_scaling_test()