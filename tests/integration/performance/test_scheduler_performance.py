"""Performance tests for the XCS schedulers and execution strategies.

This module directly tests the XCS engine's schedulers and execution strategies,
bypassing the JIT system to isolate performance characteristics of the execution
phase. It constructs graphs directly and executes them with different schedulers
to measure parallelization benefits.
"""

import logging
import statistics
import time

from ember.xcs.engine.unified_engine import execute_graph, ExecutionOptions
from ember.xcs.schedulers.unified_scheduler import NoOpScheduler, ParallelScheduler, SequentialScheduler, WaveScheduler
from ember.xcs.graph.xcs_graph import XCSGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_sequential_vs_parallel_scheduler():
    """Test the performance difference between sequential and parallel schedulers.

    This test creates a graph with multiple independent nodes, each with a sleep delay,
    and compares execution times using sequential vs parallel schedulers.
    """
    graph = XCSGraph()
    num_nodes = 10
    delay = 0.1  # seconds

    # Add nodes with sleep operators
    for i in range(num_nodes):
        node_id = f"node_{i}"
        # Simple sleep operator using a lambda
        sleep_op = lambda inputs, i=i: (
            time.sleep(delay),
            {"result": f"node_{i}_result"},
        )[1]
        graph.add_node(operator=sleep_op, node_id=node_id)

    # Test with sequential scheduler
    logger.info("\n=== Testing Sequential Scheduler ===")
    seq_scheduler = SequentialScheduler()
    seq_options = ExecutionOptions(scheduler_type="sequential")

    seq_times = []
    for i in range(3):
        start_time = time.time()
        seq_results = execute_graph(graph, {}, options=seq_options, scheduler=seq_scheduler)
        elapsed = time.time() - start_time
        seq_times.append(elapsed)
        logger.info(f"Run {i+1}: {elapsed:.4f}s")

    seq_avg = statistics.mean(seq_times)
    logger.info(f"Sequential average: {seq_avg:.4f}s")

    # Test with parallel scheduler
    logger.info("\n=== Testing Parallel Scheduler ===")
    par_scheduler = ParallelScheduler(max_workers=num_nodes)
    par_options = ExecutionOptions(scheduler_type="parallel", max_workers=num_nodes)

    par_times = []
    for i in range(3):
        start_time = time.time()
        par_results = execute_graph(graph, {}, options=par_options, scheduler=par_scheduler)
        elapsed = time.time() - start_time
        par_times.append(elapsed)
        logger.info(f"Run {i+1}: {elapsed:.4f}s")

    par_avg = statistics.mean(par_times)
    logger.info(f"Parallel average: {par_avg:.4f}s")

    # Calculate speedup
    speedup = seq_avg / par_avg if par_avg > 0 else 0

    # Report comparison
    logger.info("\n=== Performance Comparison ===")
    logger.info(f"Sequential average: {seq_avg:.4f}s")
    logger.info(f"Parallel average: {par_avg:.4f}s")
    logger.info(f"Speedup: {speedup:.2f}x")

    # Check for sufficient speedup
    theoretical_speedup = num_nodes  # In theory, we could get num_nodes times speedup
    efficiency = speedup / theoretical_speedup
    logger.info(f"Theoretical maximum speedup: {theoretical_speedup:.1f}x")
    logger.info(f"Efficiency: {efficiency:.1%} of theoretical maximum")

    # Verify parallel scheduler is significantly faster
    # The expected speedup depends on the environment; in CI it might be lower
    min_expected_speedup = 3.0
    assert (
        speedup >= min_expected_speedup
    ), f"Expected parallel scheduler to be at least {min_expected_speedup}x faster, got {speedup:.2f}x"

    # Log a warning if the speedup is suboptimal
    if speedup < 5.0:
        logger.warning(
            f"Parallel scheduler speedup ({speedup:.2f}x) is lower than ideal (5.0x) but passes the minimum threshold"
        )


def test_sequential_with_dependencies():
    """Test how parallel scheduler performs with sequential dependencies.

    This test creates a graph with a linear chain of dependencies, where
    each node depends on the previous one. In this case, parallel execution
    should provide no benefit.
    """
    graph = XCSGraph()
    num_nodes = 5
    delay = 0.1  # seconds

    # Add nodes with sleep operators in a chain
    prev_node = None
    for i in range(num_nodes):
        node_id = f"node_{i}"
        # Simple sleep operator using a lambda
        sleep_op = lambda inputs, i=i: (
            time.sleep(delay),
            {"result": f"node_{i}_result"},
        )[1]
        graph.add_node(operator=sleep_op, node_id=node_id)

        # Add edge from previous node if it exists
        if prev_node is not None:
            graph.add_edge(from_id=prev_node, to_id=node_id)

        prev_node = node_id

    # Test with sequential scheduler
    logger.info("\n=== Testing Sequential Scheduler with Dependencies ===")
    seq_scheduler = SequentialScheduler()
    seq_options = ExecutionOptions(scheduler_type="sequential")

    seq_times = []
    for i in range(3):
        start_time = time.time()
        seq_results = execute_graph(graph, {}, options=seq_options, scheduler=seq_scheduler)
        elapsed = time.time() - start_time
        seq_times.append(elapsed)
        logger.info(f"Run {i+1}: {elapsed:.4f}s")

    seq_avg = statistics.mean(seq_times)
    logger.info(f"Sequential average: {seq_avg:.4f}s")

    # Test with parallel scheduler
    logger.info("\n=== Testing Parallel Scheduler with Dependencies ===")
    par_scheduler = ParallelScheduler(max_workers=num_nodes)
    par_options = ExecutionOptions(scheduler_type="parallel", max_workers=num_nodes)

    par_times = []
    for i in range(3):
        start_time = time.time()
        par_results = execute_graph(graph, {}, options=par_options, scheduler=par_scheduler)
        elapsed = time.time() - start_time
        par_times.append(elapsed)
        logger.info(f"Run {i+1}: {elapsed:.4f}s")

    par_avg = statistics.mean(par_times)
    logger.info(f"Parallel average: {par_avg:.4f}s")

    # Calculate ratio (should be close to 1.0)
    ratio = seq_avg / par_avg if par_avg > 0 else 0

    # Report comparison
    logger.info("\n=== Performance Comparison ===")
    logger.info(f"Sequential average: {seq_avg:.4f}s")
    logger.info(f"Parallel average: {par_avg:.4f}s")
    logger.info(f"Ratio: {ratio:.2f}x")

    # Verify parallel scheduler is not significantly slower
    # Note: In CI environments, the parallel scheduler might have a bit more overhead,
    # so we allow for a more generous threshold to avoid flakiness
    assert (
        ratio >= 0.7
    ), f"Expected parallel scheduler to not be significantly slower, got {ratio:.2f}x"

    # Log a warning if the ratio is suboptimal but still passes
    if ratio < 0.9:
        logger.warning(
            f"Parallel scheduler performance suboptimal (ratio={ratio:.2f}x) but within acceptable limits"
        )


def test_diamond_pattern():
    """Test how parallel scheduler performs with a diamond dependency pattern.

    This test creates a graph with a diamond pattern:
    A → (B,C) → D
    Where B and C can be executed in parallel after A, and D requires both B and C.
    Parallel execution should provide a benefit for the middle nodes.
    """
    graph = XCSGraph()
    delay = 0.1  # seconds

    # Create nodes for the diamond pattern
    def make_sleep_op(node_name):
        return lambda inputs: (time.sleep(delay), {"result": f"{node_name}_result"})[1]

    # Add nodes
    graph.add_node(operator=make_sleep_op("A"), node_id="A")
    graph.add_node(operator=make_sleep_op("B"), node_id="B")
    graph.add_node(operator=make_sleep_op("C"), node_id="C")
    graph.add_node(operator=make_sleep_op("D"), node_id="D")

    # Add edges
    graph.add_edge(from_id="A", to_id="B")
    graph.add_edge(from_id="A", to_id="C")
    graph.add_edge(from_id="B", to_id="D")
    graph.add_edge(from_id="C", to_id="D")

    # Test with sequential scheduler
    logger.info("\n=== Testing Sequential Scheduler with Diamond Pattern ===")
    seq_scheduler = SequentialScheduler()
    seq_options = ExecutionOptions(scheduler_type="sequential")

    seq_times = []
    for i in range(3):
        start_time = time.time()
        seq_results = execute_graph(graph, {}, options=seq_options, scheduler=seq_scheduler)
        elapsed = time.time() - start_time
        seq_times.append(elapsed)
        logger.info(f"Run {i+1}: {elapsed:.4f}s")

    seq_avg = statistics.mean(seq_times)
    logger.info(f"Sequential average: {seq_avg:.4f}s")

    # Test with parallel scheduler
    logger.info("\n=== Testing Parallel Scheduler with Diamond Pattern ===")
    par_scheduler = ParallelScheduler(max_workers=4)
    par_options = ExecutionOptions(scheduler_type="parallel", max_workers=4)

    par_times = []
    for i in range(3):
        start_time = time.time()
        par_results = execute_graph(graph, {}, options=par_options, scheduler=par_scheduler)
        elapsed = time.time() - start_time
        par_times.append(elapsed)
        logger.info(f"Run {i+1}: {elapsed:.4f}s")

    par_avg = statistics.mean(par_times)
    logger.info(f"Parallel average: {par_avg:.4f}s")

    # Calculate speedup
    speedup = seq_avg / par_avg if par_avg > 0 else 0

    # Report comparison
    logger.info("\n=== Performance Comparison ===")
    logger.info(f"Sequential average: {seq_avg:.4f}s")
    logger.info(f"Parallel average: {par_avg:.4f}s")
    logger.info(f"Speedup: {speedup:.2f}x")

    # For a diamond pattern, we should see a speedup of around 1.33x
    # (4 nodes sequentially = 4 delays, vs 3 steps in parallel = 3 delays)
    # But in CI environments with threading overhead, the benefit might be smaller
    min_expected_speedup = 1.1
    assert (
        speedup >= min_expected_speedup
    ), f"Expected diamond pattern to show at least {min_expected_speedup}x speedup, got {speedup:.2f}x"

    # Log a warning if the speedup is suboptimal
    if speedup < 1.2:
        logger.warning(
            f"Diamond pattern speedup ({speedup:.2f}x) is lower than ideal (1.33x) but passes the minimum threshold"
        )


# Run tests if executed directly
if __name__ == "__main__":
    logger.info("=== Testing XCS Scheduler Performance ===")

    try:
        logger.info("\n\n=== Test 1: Independent Nodes ===")
        test_sequential_vs_parallel_scheduler()
    except Exception as e:
        logger.error(f"Error in test_sequential_vs_parallel_scheduler: {e}")

    try:
        logger.info("\n\n=== Test 2: Sequential Dependencies ===")
        test_sequential_with_dependencies()
    except Exception as e:
        logger.error(f"Error in test_sequential_with_dependencies: {e}")

    try:
        logger.info("\n\n=== Test 3: Diamond Pattern ===")
        test_diamond_pattern()
    except Exception as e:
        logger.error(f"Error in test_diamond_pattern: {e}")

    logger.info("\n=== All Tests Complete ===")
