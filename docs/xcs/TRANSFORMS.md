# XCS Transforms

XCS provides powerful transformation primitives that enable vectorization, parallelization, and distributed execution of computations. This document provides a detailed overview of the available transforms and their usage.

## Core Transform Concepts

Transforms in XCS follow these core principles:

1. **Function Transformation**: Transforms take functions as input and return transformed functions
2. **Semantic Preservation**: Transformed functions preserve the semantics of the original function
3. **Composability**: Transforms can be composed to create more complex transformations
4. **Explicit Configuration**: Transforms provide clear configuration options

## Available Transforms

### Vectorized Mapping (vmap)

The `vmap` transform enables automatic vectorization of functions to process batched inputs:

```python
from ember.api.xcs import vmap

def process_item(x):
    return x * 2

# Create a vectorized version
batch_process = vmap(process_item)

# Process multiple items at once
results = batch_process([1, 2, 3])  # [2, 4, 6]
```

#### Advanced vmap Features

**Handling Multiple Arguments**:

```python
def process_pairs(x, y):
    return x + y

# Vectorize over both arguments
vectorized = vmap(process_pairs)
result = vectorized([1, 2, 3], [10, 20, 30])  # [11, 22, 33]
```

**Specifying Axes**:

```python
# Vectorize only the first argument
vectorized = vmap(process_pairs, in_axes=(0, None))
result = vectorized([1, 2, 3], 10)  # [11, 12, 13]
```

**Nested Batching**:

```python
# Vectorize in two dimensions
double_vectorized = vmap(vmap(process_item))
result = double_vectorized([[1, 2], [3, 4]])  # [[2, 4], [6, 8]]
```

### Parallel Mapping (pmap)

The `pmap` transform enables parallel execution across multiple cores:

```python
from ember.api.xcs import pmap
import time

def slow_computation(x):
    time.sleep(1)  # Simulate work
    return x * 2

# Create a parallelized version
parallel_process = pmap(slow_computation)

# Process items in parallel (much faster than sequential)
results = parallel_process([1, 2, 3, 4])  # [2, 4, 6, 8]
```

#### Advanced pmap Features

**Controlling Worker Count**:

```python
# Specify number of worker threads
parallel_process = pmap(slow_computation, num_workers=4)
```

**Error Handling**:

```python
from ember.api.xcs import TransformOptions

# Configure error handling
options = TransformOptions(propagate_errors=True)
parallel_process = pmap(risky_function, options=options)
```

**Timeouts**:

```python
# Add timeout to prevent hanging
options = TransformOptions(timeout=5.0)
parallel_process = pmap(slow_computation, options=options)
```

### Mesh Sharding (mesh_sharded)

The `mesh_sharded` transform enables distributed execution across a device mesh:

```python
from ember.api.xcs import mesh_sharded, DeviceMesh, PartitionSpec

# Create a 2D device mesh
mesh = DeviceMesh(shape=(2, 2))

# Specify how to partition the data
pspec = PartitionSpec(0, 1)  # Partition along both dimensions

# Create a sharded function
sharded_fn = mesh_sharded(heavy_computation, mesh=mesh, partition_spec=pspec)

# Execute with data distributed across the mesh
result = sharded_fn(large_data)
```

#### Advanced Mesh Features

**Custom Device Specification**:

```python
# Specify devices explicitly
mesh = DeviceMesh(
    devices=["gpu:0", "gpu:1", "cpu:0", "cpu:1"],
    shape=(2, 2)
)
```

**Partial Sharding**:

```python
# Only shard along first dimension
pspec = PartitionSpec(0, None)
```

**Nested Meshes**:

```python
# Create hierarchical meshes
outer_mesh = DeviceMesh(shape=(2,))
inner_mesh = DeviceMesh(shape=(2,))

# Compose sharding transformations
outer_sharded = mesh_sharded(inner_sharded_fn, mesh=outer_mesh, partition_spec=PartitionSpec(0))
```

## Combining Transforms

Transforms can be combined to create more powerful transformations:

```python
# Vectorize and parallelize
vmap_then_pmap = pmap(vmap(function))

# Process batches in parallel
result = vmap_then_pmap(batched_data)
```

### Transformation Order

The order of transforms matters:

- `pmap(vmap(f))`: Each worker processes a batch (often more efficient)
- `vmap(pmap(f))`: Each element processed in parallel (higher overhead, more parallelism)

### Integration with JIT System

Transforms can be combined with Ember's JIT system for additional optimizations:

```python
from ember.api.xcs import jit, vmap

# JIT-compiled vectorized function
@jit
def process_item(x):
    return expensive_computation(x)

# Vectorized version with JIT optimization
batch_process = vmap(process_item)
```

The relationship between transforms and the different JIT approaches (jit, structural_jit, autograph) is described in more detail in [JIT Overview](JIT_OVERVIEW.md).

## Transform Implementation Details

### vmap Implementation

The `vmap` transform works by:

1. Splitting the input batch into individual elements
2. Applying the original function to each element
3. Combining the results into a batched output

For optimization, it:
- Uses vectorized operations when available
- Employs batch processing primitives
- Handles nested data structures correctly

### pmap Implementation

The `pmap` transform works by:

1. Creating a thread pool of worker threads
2. Dividing work among workers
3. Executing the function on each worker
4. Combining results in the original order

It automatically handles:
- Thread creation and management
- Work distribution
- Result aggregation
- Error propagation

### mesh_sharded Implementation

The `mesh_sharded` transform works by:

1. Partitioning input data according to the partition spec
2. Mapping partitions to devices in the mesh
3. Executing the function on each device with its partition
4. Gathering and combining the results

## Best Practices

1. **Choose the Right Transform**: Use `vmap` for vectorization, `pmap` for CPU parallelism, and `mesh_sharded` for distributed execution
2. **Consider Grain Size**: Ensure each work unit is substantial enough to justify parallelization overhead
3. **Tune Worker Count**: Adjust `num_workers` based on your workload and hardware
4. **Use Appropriate Axes**: Configure `in_axes` and `out_axes` to match your data structure
5. **Combine Transforms Wisely**: Consider the trade-offs when composing transforms