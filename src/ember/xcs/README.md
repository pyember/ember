# Ember XCS: High-Performance Execution Framework

The Ember XCS (Accelerated Compound Systems) module provides a high-performance distributed
execution framework for computational graphs. It implements a directed acyclic graph (DAG)
architecture for operator composition, intelligent scheduling, and just-in-time tracing.

## Architecture

XCS follows a clean, modular architecture with well-defined layers:

1. **Facade Layer** (`__init__.py`): Providing a simple, unified API that abstracts implementation details
2. **Component Layer**:
   - `tracer/`: JIT compilation and graph building mechanisms
   - `engine/`: Execution scheduling and runtime optimization
   - `transforms/`: Function transformations (vectorization, parallelization)
   - `graph/`: Computational graph representation and manipulation
   - `utils/`: Shared utilities and helper functions

## Key Components

### JIT Tracing and Optimization

The JIT (Just-In-Time) tracing system enables automatic optimization of operator execution
by analyzing operator structure and execution patterns:

```python
from ember.xcs import jit, structural_jit

# Basic JIT compilation
@jit
class SimpleOperator(Operator):
    def forward(self, *, inputs):
        # Simple processing logic
        return {"result": process(inputs["query"])}

# Advanced structural optimization for complex operators
@structural_jit(execution_strategy="auto", parallel_threshold=3)
class CompositeOperator(Operator):
    def __init__(self):
        self.op1 = SubOperator1()
        self.op2 = SubOperator2()
        self.op3 = SubOperator3()
    
    def forward(self, *, inputs):
        # Multi-stage execution with automatic parallelization
        stage1 = self.op1(inputs=inputs)
        stage2 = self.op2(inputs=stage1)
        return self.op3(inputs=stage2)
```

### Automatic Graph Building and Execution

Building execution graphs automatically from traced execution, enabling optimization
and parallel execution:

```python
from ember.xcs import autograph, execute, execution_options

# Recording operations into a graph
with autograph() as graph:
    # Operations are recorded, not executed
    x = op1(inputs={"query": "Example"})
    y = op2(inputs=x)
    z = op3(inputs=y)
    
# Configuring execution parameters
opts = execution_options(
    max_workers=4,    # Using 4 worker threads
    scheduler="parallel"  # Enabling parallel execution
)

# Executing the graph with optimization
results = execute(graph, options=opts)
print(f"Final result: {results['z']}")

# Selective execution of specific nodes
subset_results = execute(graph, output_nodes=["y"])
```

### Function Transformations

The transform system provides high-level operations for batching, parallelization, and
distributed execution:

```python
from ember.xcs import vmap, pmap, mesh_sharded
from ember.xcs import DeviceMesh, PartitionSpec

# Creating a simple function to transform
def process_item(item):
    return {"processed": transform(item["data"]), "id": item["id"]}

# Vectorizing for batch processing
batch_fn = vmap(process_item)
batch_results = batch_fn(inputs={
    "data": ["item1", "item2", "item3"],
    "id": [1, 2, 3],
    "options": {"format": "json"}  # Non-batched param applied to all items
})
# batch_results == {"processed": ["ITEM1", "ITEM2", "ITEM3"], "id": [1, 2, 3]}

# Parallelizing for multi-core execution
parallel_fn = pmap(process_item)
parallel_results = parallel_fn(inputs={"data": ["a", "b", "c"], "id": [1, 2, 3]})

# Setting up distributed mesh execution
devices = [0, 1, 2, 3]  # Available compute devices
mesh = DeviceMesh(devices=devices, mesh_shape=(2, 2))
pspec = PartitionSpec(0, 1)  # Partition specification

# Creating a distributed function
sharded_fn = mesh_sharded(process_item, mesh, pspec)
distributed_results = sharded_fn(inputs=large_dataset)

# Combining transforms for nested parallelism
distributed_batch_fn = pmap(vmap(process_item))
```

## Advanced API Design

The XCS module provides both simple direct interfaces and configurable advanced options:

```python
# Core functionality through clean imports
from ember.xcs import jit, vmap, pmap, autograph, execute

# Advanced configuration through explicit API
from ember.xcs.api.types import JITOptions, XCSExecutionOptions
from ember.xcs.api.core import XCSAPI

# Creating customized API instance
xcs = XCSAPI()

# Using with detailed configuration
@xcs.jit(options=JITOptions(
    sample_input={"query": "test"},
    force_trace=False,
    recursive=True
))
class OptimizedOperator(Operator):
    # Implementation details...
    pass
```

## Error Handling and Fallbacks

The XCS module includes robust error handling with graceful fallbacks for testing environments:

- Automatic fallback to stub implementations when dependencies aren't available
- Clear warning messages with detailed diagnostics
- Type-safe interfaces with runtime protocol checks
- Testing helpers for isolating functionality

```python
# Testing with JIT disabled
from ember.xcs.tracer.structural_jit import disable_structural_jit

with disable_structural_jit():
    # Code here runs without optimization for comparison/testing
    result = my_operator(inputs=test_input)
```

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

## Real-World Examples

### Optimizing a Multi-Stage Operator Pipeline

```python
from ember.xcs import jit, autograph, execute

@jit
class TextProcessor(Operator):
    def __init__(self):
        self.parser = ParserOperator()
        self.analyzer = AnalyzerOperator()
        self.transformer = TransformerOperator()
        self.formatter = FormatterOperator()
    
    def forward(self, *, inputs):
        # Automatically optimized multi-stage pipeline
        parsed = self.parser(inputs=inputs)
        analyzed = self.analyzer(inputs=parsed)
        transformed = self.transformer(inputs=analyzed)
        return self.formatter(inputs=transformed)

# Creating and using the optimized operator
processor = TextProcessor()
result = processor(inputs={"text": "Process this text"})
```

### Batch Processing with Vectorization

```python
from ember.xcs import vmap

def calculate_metrics(item):
    """Processing a single data item."""
    return {
        "id": item["id"],
        "score": compute_score(item["data"]),
        "normalized": normalize(item["data"]),
        "timestamp": get_current_time()
    }

# Creating vectorized version for batch processing
batch_calculator = vmap(calculate_metrics)

# Processing entire dataset at once
dataset = {
    "id": [101, 102, 103, 104, 105],
    "data": [item1, item2, item3, item4, item5],
    "metadata": global_metadata  # Applied to all items
}

# Efficient parallel processing of all items
results = batch_calculator(inputs=dataset)
```

For more detailed information, see the documentation files and examples in the project repository.