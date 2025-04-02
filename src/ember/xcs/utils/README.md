# Ember XCS Execution Framework

## Overview

The XCS execution framework provides a simplified, high-performance system for parallel execution of operations in Ember computational graphs. Designed with principles from engineering giants like Jeff Dean and Sanjay Ghemawat, it offers minimal abstractions with maximum utility.

## Key Components

### `Executor` Protocol

Defines the minimal interface for executing batched tasks:

```python
@runtime_checkable
class Executor(Protocol[T, U]):
    def execute(self, fn: Callable[[T], U], inputs: List[T]) -> List[U]:
        ...
```

### Concrete Executors

1. `ThreadExecutor` - Optimized for standard Ember workloads, including most model operations, data transformations, and mixed computation tasks. Best for moderate concurrency levels and operations that involve both API calls and local computation.

2. `AsyncExecutor` - Specialized for high-concurrency LLM API operations, particularly useful for ensemble operations, parallel model calls across providers, and scenarios with many simultaneous model API requests.

### `Dispatcher`

Smart execution coordinator that:
- Analyzes function characteristics to determine optimal execution method
- Routes operations to the most efficient executor
- Handles error policies and resource management
- Tracks execution metrics for adaptive optimization

## Usage in Ember

```python
# Direct use in Operators
from ember.xcs.utils.executor import Dispatcher

class ParallelOperator(Operator[Input, Output]):
    def forward(self, *, inputs: Input) -> Output:
        # Create dispatcher with configuration
        dispatcher = Dispatcher(
            max_workers=4,      # Control concurrency
            fail_fast=False,    # Continue on errors
            executor="auto"     # Auto-select optimal executor
        )
        
        # Process batched inputs with optimal parallelism
        results = dispatcher.map(self._process_item, inputs_list)
        return self._combine_results(results)

# Within transforms
from ember.xcs.transforms import vmap

@vmap(parallel=True, max_workers=4)  # Uses Dispatcher internally
def vectorized_function(*, inputs):
    # Automatically parallelized across inputs
    return {"result": process(inputs["data"])}
```

## When to Use Each Executor

| Workload Type | Recommended Executor | Example |
|---------------|---------------------|---------|
| Standard model operations | `ThreadExecutor` | Single model completions, data loading |
| Mixed computation & I/O | `ThreadExecutor` | Data transformation with API calls |
| Many concurrent API calls | `AsyncExecutor` | Multi-model ensemble, parallel evals |
| High-concurrency operations | `AsyncExecutor` | Batched generation across providers |
| Unknown/general purpose | `Dispatcher` with `"auto"` | Let the system decide |

## Error Handling

Set `fail_fast=False` when operations can tolerate partial failures, particularly useful for:
- Evaluation across many examples where some may fail
- Ensemble operations where individual model failures shouldn't stop the process
- Speculative execution patterns where some branches may fail

## Performance Considerations

- Default concurrency levels are automatically set based on system resources
- For I/O-heavy workloads, increase `max_workers` significantly (20-100+)
- For CPU-bound operations, keep `max_workers` close to available core count
- Use `timeout` parameter to prevent long-running operations from blocking execution

## Integration with XCS

The execution framework integrates seamlessly with other XCS components:
- Transforms (`vmap`, `pmap`) use it to parallelize operations
- JIT compilation uses performance metrics to optimize execution plans
- Schedulers coordinate execution across the computational graph