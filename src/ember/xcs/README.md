# Ember XCS: Unified Execution Framework

XCS (Accelerated Compound Systems) provides a high-performance distributed execution framework for computational graphs. It implements a directed acyclic graph (DAG) architecture for operator composition, intelligent scheduling, and just-in-time compilation.

## Architecture

XCS follows a clean, unified architecture with stratified layers:

1. **Protocol Layer**: Core interfaces defining component contracts
2. **Strategy Layer**: Pluggable strategy implementations for each component
3. **Implementation Layer**: Concrete implementations with consistent interfaces
4. **Facade Layer**: Simplified public API abstracting implementation details

## Key Components

### JIT Compilation

The JIT system combines multiple compilation strategies under a consistent interface:

```python
from ember.xcs import jit

# Simple usage with automatic strategy selection
@jit
class MyOperator(Operator):
    def forward(self, *, inputs):
        return {"result": process(inputs["data"])}

# Parameterized usage with explicit strategy
@jit(mode="enhanced", sample_input={"query": "example"})
class CompositeOperator(Operator):
    def __init__(self):
        self.op1 = SubOperator1()
        self.op2 = SubOperator2()
    
    def forward(self, *, inputs):
        intermediate = self.op1(inputs=inputs)
        return self.op2(inputs=intermediate)
```

### Dependency Analysis

Provides unified dependency tracking and analysis for all graph operations:

- Efficient transitive closure calculation
- Topological sorting with cycle detection
- Execution wave calculation for parallel scheduling

### Execution Scheduling

Unified scheduler implementations share a common interface and strategy pattern:

```python
from ember.xcs import jit, execution_options, create_scheduler

# Using the JIT decorator with explicit execution options
@jit
class MyOperator(Operator):
    def forward(self, *, inputs):
        return {"result": process(inputs["data"])}

# Create an instance of the operator
op = MyOperator()

# Control execution with context manager
with execution_options(scheduler="wave", max_workers=4):
    results = op(inputs={"query": "example"})
    
# Or create a custom scheduler directly
scheduler = create_scheduler("parallel", max_workers=8)
```

### Function Transformations

High-level operations for batching and parallelization:

```python
from ember.xcs import vmap, pmap, pjit, compose

# Vectorizing a function for batch processing
batch_process = vmap(process_item)
batch_results = batch_process(inputs={"data": ["item1", "item2", "item3"]})

# Parallelizing execution across multiple workers
parallel_process = pmap(process_item, num_workers=4)
parallel_results = parallel_process(inputs={"data": large_dataset})

# Combining transformations
vectorized_parallel = compose(
    vmap(batch_size=32), 
    pmap(num_workers=4)
)
optimized_fn = vectorized_parallel(process_item)

# Using combined JIT+parallel transformation
optimized_fn = pjit(process_item, mode="enhanced")
```

## Architectural Components

XCS is organized into the following key packages:

- **jit/**: JIT compilation system with pluggable strategies
  - **strategies/**: Different JIT compilation approaches (trace, structural, enhanced)
  - **core.py**: Main JIT decorator implementation
  - **cache.py**: Caching mechanism for compiled functions

- **schedulers/**: Unified execution scheduler system
  - **base_scheduler.py**: Core scheduler interface
  - **unified_scheduler.py**: Concrete scheduler implementations
  - **factory.py**: Factory for creating appropriate schedulers

- **graph/**: Graph representation and dependency analysis 
  - **xcs_graph.py**: Core graph data structure
  - **dependency_analyzer.py**: Dependency tracking and analysis
  - **graph_builder.py**: Graph construction from traces

- **engine/**: Unified execution engine
  - **unified_engine.py**: Core execution functionality
  - **execution_options.py**: Execution configuration

- **transforms/**: Function transformations
  - **transform_base.py**: Shared foundation for all transforms
  - **vmap.py**: Vectorization implementation
  - **pmap.py**: Parallelization implementation
  - **mesh.py**: Device mesh-based sharding

- **common/**: Shared data structures
  - **plans.py**: Execution plan representations

- **tracer/**: Tracing infrastructure
  - **xcs_tracing.py**: Core tracing functionality
  - **autograph.py**: Automatic graph building

## Extension Points

XCS is designed for extensibility via clearly defined protocols:

- Create custom schedulers by implementing `BaseScheduler` or extending `BaseSchedulerImpl`
- Add new JIT strategies by implementing the `Strategy` protocol
- Implement custom graph transformations by extending `BaseTransformation`
- Define custom execution policies using the execution options system

For more examples, see the `examples/` directory.