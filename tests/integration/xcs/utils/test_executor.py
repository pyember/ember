"""
Test script for the simplified executor framework.
"""

import time
from ember.xcs.utils.executor import Dispatcher, ThreadExecutor, AsyncExecutor


def simple_function(*, inputs):
    """Simple test function that just returns the input."""
    return {"result": inputs["value"] * 2}


def io_bound_function(*, inputs):
    """Simulated I/O bound function with sleep."""
    time.sleep(0.1)  # Simulate I/O operation
    return {"result": f"processed-{inputs['value']}"}


def test_thread_executor():
    """Test explicit thread executor."""
    print("\nTesting ThreadExecutor...")
    executor = ThreadExecutor(
        max_workers=4,
        timeout=None,
        fail_fast=True
    )
    
    try:
        inputs = [{"value": i} for i in range(10)]
        results = executor.execute(simple_function, inputs)
        print(f"Results: {[r['result'] for r in results]}")
    finally:
        executor.close()


def test_async_executor():
    """Test explicit async executor."""
    print("\nTesting AsyncExecutor...")
    executor = AsyncExecutor(
        max_concurrency=4,
        timeout=None,
        fail_fast=True
    )
    
    try:
        inputs = [{"value": f"item-{i}"} for i in range(10)]
        results = executor.execute(io_bound_function, inputs)
        print(f"Results: {[r['result'] for r in results]}")
    finally:
        executor.close()


def test_dispatcher_auto():
    """Test dispatcher with auto executor selection."""
    print("\nTesting Dispatcher with auto selection...")
    dispatcher = Dispatcher(max_workers=4)
    
    try:
        # Should select thread for CPU-bound simple_function
        inputs = [{"value": i} for i in range(5)]
        results = dispatcher.map(simple_function, inputs)
        print(f"CPU-bound results: {[r['result'] for r in results]}")
        
        # Should select async for I/O-bound io_bound_function
        inputs = [{"value": f"item-{i}"} for i in range(5)]
        results = dispatcher.map(io_bound_function, inputs)
        print(f"I/O-bound results: {[r['result'] for r in results]}")
    finally:
        dispatcher.close()


def test_error_handling():
    """Test error handling settings."""
    def error_function(*, inputs):
        if inputs["value"] % 3 == 0:
            raise ValueError(f"Error for input {inputs['value']}")
        return {"result": inputs["value"] * 2}
    
    print("\nTesting error handling - fail_fast=True...")
    dispatcher_fail_fast = Dispatcher(
        max_workers=4,
        fail_fast=True
    )
    
    try:
        inputs = [{"value": i} for i in range(10)]
        try:
            results = dispatcher_fail_fast.map(error_function, inputs)
            print("Should have failed but didn't!")
        except Exception as e:
            print(f"Got expected exception: {e}")
    finally:
        dispatcher_fail_fast.close()
    
    print("\nTesting error handling - fail_fast=False...")
    dispatcher_continue = Dispatcher(
        max_workers=4,
        fail_fast=False
    )
    
    try:
        inputs = [{"value": i} for i in range(10)]
        results = dispatcher_continue.map(error_function, inputs)
        print(f"Results with continue: {results}")
        print(f"Number of results: {len(results)}")
    finally:
        dispatcher_continue.close()


if __name__ == "__main__":
    test_thread_executor()
    test_async_executor()
    test_dispatcher_auto()
    test_error_handling()