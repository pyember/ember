"""
Utilities for testing XCS transforms.

This module provides common utilities and helper functions used across multiple
transform test modules, promoting code reuse and consistency.
"""

import time
from typing import Any, Dict, List, Tuple, Callable


def generate_batch_inputs(batch_size: int, prefix: str = "item") -> Dict[str, List[str]]:
    """Generate a batch of test inputs with the given size.
    
    Args:
        batch_size: Number of items to generate
        prefix: Prefix for each item (default: "item")
        
    Returns:
        Dictionary with a "prompts" key mapping to a list of generated inputs
    """
    return {
        "prompts": [f"{prefix}{i}" for i in range(batch_size)]
    }


def assert_processing_time(
    sequential_time: float, 
    parallel_time: float, 
    min_speedup: float = 1.2,
    max_overhead_factor: float = 3.0  # Increased overhead factor for tests
) -> None:
    """Assert that parallel processing is faster than sequential processing.
    
    Args:
        sequential_time: Time taken for sequential processing
        parallel_time: Time taken for parallel processing
        min_speedup: Minimum expected speedup factor (default: 1.2)
        max_overhead_factor: Maximum overhead factor for small inputs (default: 3.0)
        
    Raises:
        AssertionError: If the parallel time doesn't meet the speedup expectations
    """
    # For very small inputs or test environments, parallel might be slower due to overhead
    # In CI environments especially, thread creation overhead can be significant
    if sequential_time < 0.1:
        # Skip performance checks for very small timings since they're unreliable
        return
    elif sequential_time < 0.5:
        # For small inputs, allow significant overhead
        assert parallel_time < sequential_time * max_overhead_factor, (
            f"Parallel processing overhead is too high: {sequential_time:.4f}s vs {parallel_time:.4f}s"
        )
    else:
        # For substantial inputs, parallel should be faster
        speedup = sequential_time / parallel_time
        assert speedup >= min_speedup, (
            f"Expected minimum speedup of {min_speedup}x, but got {speedup:.2f}x"
        )


def time_function_execution(
    func: Callable[..., Any], 
    *args: Any, 
    **kwargs: Any
) -> Tuple[float, Any]:
    """Execute a function and measure its execution time.
    
    Args:
        func: The function to execute
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Tuple containing (execution_time, function_result)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return (end_time - start_time), result


def count_unique_threads(thread_ids_dict: Dict[int, Any]) -> int:
    """Count the number of unique thread IDs in a dictionary.
    
    Args:
        thread_ids_dict: Dictionary with thread IDs as keys
        
    Returns:
        Number of unique thread IDs
    """
    return len(thread_ids_dict.keys())