# Ember XCS: High-Performance Execution Framework

The Ember XCS (eXecutable Computation System) module provides a high-performance distributed
execution framework for computational graphs. It implements a directed acyclic graph (DAG)
architecture for operator composition, intelligent scheduling, and just-in-time tracing.

## Architecture

XCS follows a clean, modular architecture with well-defined layers:

1. **Facade Layer** (`__init__.py`): Provides a simple, unified API that abstracts implementation details
2. **Component Layer**:
   - `tracer/`: JIT compilation and graph building
   - `engine/`: Execution scheduling and runtime
   - `transforms/`: Function transformations (vectorization, parallelization)
   - `graph/`: Computational graph representation
   - `utils/`: Shared utilities

The implementation adheres to SOLID principles:
- **Single Responsibility**: Each module has a specific focus
- **Open/Closed**: Extensible through protocols and well-defined interfaces
- **Liskov Substitution**: Implementations are interchangeable
- **Interface Segregation**: Clean, minimal interfaces
- **Dependency Inversion**: Depends on abstractions, not implementations

## Key Components

### JIT Tracing

The JIT (Just-In-Time) tracing system enables automatic optimization of operator execution:

```python
from ember.xcs import jit

@jit
class MyOperator(Operator):
    def forward(self, *, inputs):
        # Complex multi-step calculation
        return result
```

### Automatic Graph Building

Build execution graphs automatically from traced execution:

```python
from ember.xcs import autograph, execute

with autograph() as graph:
    # Operations are recorded, not executed
    result = my_op(inputs={"query": "Example"})
    
# Execute the graph
results = execute(graph)
```

### Function Transformations

The transform system provides high-level operations for batching and parallelization:

```python
from ember.xcs import vmap, pmap, mesh_sharded

# Vectorize a function
batch_fn = vmap(my_function)

# Parallelize a function
parallel_fn = pmap(my_function)

# Distributed mesh execution
sharded_fn = mesh_sharded(my_function)
```

## Simple API Design

The XCS module provides a simplified, intuitive interface to complex functionality:

```python
# Clean imports
from ember.xcs import jit, vmap, pmap, autograph, execute

# Access to the unified API for advanced configuration
from ember.xcs import XCSGraph, ExecutionOptions
```

## Error Handling

The XCS module includes robust error handling with graceful fallbacks for testing environments:

- Automatic fallback to stub implementations when dependencies aren't available
- Clear warning messages for diagnostic purposes
- Type-safe interfaces with runtime protocol checks

## Usage in Project

XCS is accessible through two main import paths:

1. **Direct imports** for power users:
   ```python
   from ember.xcs import jit, vmap, pmap
   ```

2. **API facade** for simplified access:
   ```python
   from ember.api.xcs import jit, vmap, pmap
   ```

For more information, see the [documentation](https://docs.pyember.org).