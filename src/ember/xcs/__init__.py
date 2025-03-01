# XCS Unified Pipeline
# All new DAG-based processing, scheduling, and JIT/tracing use XCSGraph, XCSScheduler,
# and the unified XCS tracing in xcs_tracing.py.

# Note: To avoid circular imports, import the components directly in your code:
# from ember.xcs.engine.xcs_engine import execute_graph
# from ember.xcs.engine.execution_options import ExecutionOptions, execution_options
# from ember.xcs.tracer.tracer_decorator import jit
