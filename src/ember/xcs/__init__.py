"""
XCS: Unified Execution Pipeline

The XCS (eXecutable Computation System) module provides a high-performance distributed
execution framework for computational graphs. It implements a directed acyclic graph (DAG)
architecture for operator composition, intelligent scheduling, and just-in-time tracing.

Key components:
- Graph: DAG-based intermediate representation (IR) for defining operator pipelines
- Engine: Concurrent execution scheduler with automatic dependency resolution
- Tracer: JIT tracing system that creates execution graphs from function calls
- Transforms: Higher-order operations for batching, parallelization, and more

Usage:
To avoid circular imports, import the components directly in your code:

```python
from ember.xcs.engine.xcs_engine import execute_graph
from ember.xcs.engine.execution_options import ExecutionOptions, execution_options
from ember.xcs.tracer.tracer_decorator import jit
```
"""
