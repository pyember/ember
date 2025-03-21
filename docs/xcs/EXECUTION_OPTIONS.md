# Execution Options Guide

## Overview

The execution options system in Ember XCS allows you to control how operations are executed, including parallelism settings, device selection, and scheduling strategies. This guide explains how to use execution options effectively in your applications.

## Basic Usage

Execution options can be applied in two ways:

1. As a temporary context using the `execution_options` context manager
2. As a global setting using `set_execution_options`

### Context Manager (Recommended)

Using the context manager is the recommended approach for most use cases, as it ensures options are only applied within a specific scope and automatically restored afterward:

```python
from ember.xcs.engine.execution_options import execution_options

# Run with parallel execution and 4 workers
with execution_options(use_parallel=True, max_workers=4):
    result = vectorized_op(inputs={"prompt": prompt, "seed": seeds})
    
# Run with sequential execution
with execution_options(use_parallel=False):
    result = vectorized_op(inputs={"prompt": prompt, "seed": seeds})
```

### Global Settings

For cases where you want to set execution options for an entire application, you can use the `set_execution_options` function:

```python
from ember.xcs.engine.execution_options import set_execution_options

# Set global execution options
set_execution_options(use_parallel=True, max_workers=8)
```

You can retrieve the current execution options with `get_execution_options()`.

## Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_parallel` | `bool` | `True` | Whether to use parallel execution where possible |
| `max_workers` | `Optional[int]` | `None` | Maximum number of worker threads for parallel execution |
| `device_strategy` | `str` | `"auto"` | Strategy for device selection ('auto', 'cpu', 'gpu', etc.) |
| `enable_caching` | `bool` | `False` | Whether to cache intermediate results |
| `trace_execution` | `bool` | `False` | Whether to trace execution for debugging |
| `timeout_seconds` | `Optional[float]` | `None` | Maximum execution time in seconds before timeout |
| `scheduler` | `Optional[str]` | `None` | Legacy parameter for backward compatibility |

## Usage with Transforms

### vmap

Execution options are particularly useful when combined with vmap for controlling how batched operations are processed:

```python
from ember.xcs.transforms.vmap import vmap
from ember.xcs.engine.execution_options import execution_options

# Create vectorized version of an operator
vectorized_op = vmap(my_operator)

# Run with parallel execution
with execution_options(use_parallel=True, max_workers=4):
    batch_result = vectorized_op(inputs={"prompt": prompt, "seed": seeds})
```

### pmap

For pmap, execution options control the underlying execution behavior:

```python
from ember.xcs.transforms.pmap import pmap
from ember.xcs.engine.execution_options import execution_options

# Create parallelized version with specific worker count
parallelized_op = pmap(my_operator, num_workers=4)

# execution_options can still affect other aspects of execution
with execution_options(timeout_seconds=10.0):
    result = parallelized_op(inputs=input_data)
```

## Debugging and Performance Profiling

You can enable tracing for debugging and performance profiling:

```python
with execution_options(trace_execution=True):
    result = complex_operator(inputs=input_data)
```

## Best Practices

1. **Prefer the context manager** over global settings to limit the scope of changes.

2. **Match workers to workload and hardware**:  
   Set `max_workers` based on your workload characteristics and available CPU cores. A good starting point is the number of physical cores for compute-bound tasks, or more for I/O-bound tasks.

3. **Consider caching for repeated operations**:  
   Enable caching when the same operation is performed multiple times with identical inputs.

4. **Set appropriate timeouts**:  
   Use `timeout_seconds` to prevent operations from running indefinitely, especially in production environments.

5. **Benchmark different settings**:  
   Test different combinations of options to find the optimal configuration for your specific use case.

## Legacy Support

For compatibility with older code, the `scheduler="sequential"` option is supported and will automatically set `use_parallel=False`. This ensures older code continues to work without modification.

```python
# Legacy approach (still works)
with execution_options(scheduler="sequential"):
    result = vectorized_op(inputs={"prompt": prompt, "seed": seeds})
    
# Modern approach (preferred)
with execution_options(use_parallel=False):
    result = vectorized_op(inputs={"prompt": prompt, "seed": seeds})
```

## Exception Handling

Invalid option names will raise a `ValueError` with a message indicating the invalid option. This helps catch configuration errors early in development:

```python
try:
    with execution_options(invalid_option=True):
        result = op(inputs=data)
except ValueError as e:
    print(f"Configuration error: {e}")  # Output: Configuration error: Invalid execution option: invalid_option
```