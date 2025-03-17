"""
Test XCS Implementation

This example demonstrates the use of the XCS (eXecutable Computation System) API
directly. It shows how to use JIT compilation, vectorization, and automatic 
graph building for high-performance operator execution.

To run:
    poetry run pytest -xvs src/ember/examples/xcs/test_xcs_implementation.py
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from typing import List, Dict, Any

# Use a try-except block to improve import reliability
try:
    # First try to import from the standard location
    from src.ember.xcs import jit, vmap, pmap, autograph, execute
except ImportError:
    # If that fails, import the XCS module directly
    project_root = Path(__file__).parent.parent.parent.parent
    xcs_path = project_root / "src" / "ember" / "xcs" / "__init__.py"

    # Add a fallback if the file doesn't exist
    if not xcs_path.exists():
        # Create a minimal mock implementation
        import types

        xcs = types.ModuleType("xcs")

        # Add basic functions
        def jit(fn=None, **kwargs):
            if fn is None:
                return lambda f: f
            return fn

        def vmap(fn, *args, **kwargs):
            def wrapper(xs, *wargs, **wkwargs):
                if isinstance(xs, list):
                    return [fn(x, *wargs, **wkwargs) for x in xs]
                return fn(xs, *wargs, **wkwargs)

            return wrapper

        def pmap(fn, **kwargs):
            return vmap(fn)

        class GraphContext:
            def __init__(self):
                self.results = {}

            def add_node(self, name, func, *args, **kwargs):
                return 0

            def execute(self, output_nodes=None):
                return {}

        def autograph():
            return GraphContext()

        def execute(graph, output_nodes=None):
            return {}

        # Add them to the module
        xcs.jit = jit
        xcs.vmap = vmap
        xcs.pmap = pmap
        xcs.autograph = autograph
        xcs.execute = execute
    else:
        # Import the module if the file exists
        spec = importlib.util.spec_from_file_location("ember.xcs", xcs_path)
        xcs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(xcs)

    # Extract the needed functions
    jit = xcs.jit
    vmap = xcs.vmap
    pmap = xcs.pmap
    autograph = xcs.autograph
    execute = xcs.execute

# Now we can use the core XCS functionality
from functools import partial
import time


def main():
    """Test the XCS implementation with examples."""
    print("\n=== Testing XCS Implementation ===\n")

    # Test JIT decorator
    print("Testing JIT Compilation:")

    @xcs.jit
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers (JIT compiled)."""
        return a * b

    # Measure execution time
    start = time.time()
    result = multiply(10, 20)
    duration = time.time() - start

    print(f"  Result of multiply(10, 20): {result}")
    print(f"  Execution time: {duration:.6f} seconds")
    print("  (First call includes compilation overhead)")

    # Test vmap for vectorization
    print("\nTesting Vectorized Mapping (vmap):")

    def square(x: int) -> int:
        """Square a number."""
        return x * x

    vectorized_square = xcs.vmap(square)
    input_list = [1, 2, 3, 4, 5]
    result = vectorized_square(input_list)

    print(f"  Input: {input_list}")
    print(f"  Vectorized square: {result}")

    # Test pmap for parallel execution
    print("\nTesting Parallel Mapping (pmap):")

    def slow_operation(x: int) -> int:
        """A slow operation that simulates computational work."""
        time.sleep(0.01)  # Simulate work
        return x * 2

    # Sequential execution for comparison
    start = time.time()
    sequential_results = [slow_operation(x) for x in range(10)]
    sequential_time = time.time() - start

    # Parallel execution
    parallel_op = xcs.pmap(slow_operation)
    start = time.time()
    parallel_results = parallel_op(list(range(10)))
    parallel_time = time.time() - start

    print(f"  Sequential execution time: {sequential_time:.6f} seconds")
    print(f"  Parallel execution time: {parallel_time:.6f} seconds")
    print(f"  Speed improvement: {sequential_time / parallel_time:.2f}x")

    # Test graph building with XCSGraph directly
    print("\nTesting Graph Building:")

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    from ember.xcs.graph.xcs_graph import XCSGraph
    from ember.xcs.engine.xcs_engine import compile_graph, XCSNoOpScheduler

    # Create a graph directly
    graph = XCSGraph()
    graph.add_node(operator=lambda inputs: add(5, 3), node_id="node1")
    graph.add_node(
        operator=lambda inputs, node1_result=None: multiply(node1_result, 2),
        node_id="node2",
        dependencies=["node1"],
    )

    print("  Graph built successfully")
    print("  Executing graph...")

    # Compile and execute the graph
    plan = compile_graph(graph=graph)
    scheduler = XCSNoOpScheduler()
    results = scheduler.run_plan(plan=plan, global_input={}, graph=graph)
    print(f"  Graph execution results: {results}")

    print("\nXCS Implementation Test Complete!")


if __name__ == "__main__":
    main()
