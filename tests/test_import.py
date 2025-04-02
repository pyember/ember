"""Test basic imports after refactoring."""

import sys
import os
import inspect

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Now import the modules to verify they still work
from ember.xcs.jit import jit, JITMode
from ember.xcs.transforms import vmap, pmap
from ember.xcs.engine.unified_engine import execute_graph, ExecutionOptions
from ember.xcs.schedulers.unified_scheduler import (
    SequentialScheduler, 
    ParallelScheduler
)

# Print basic info to confirm imports work
print("JIT modes:", [mode.value for mode in JITMode])
print("JIT function signature:", inspect.signature(jit))
print("vmap function signature:", inspect.signature(vmap))
print("execute_graph function signature:", inspect.signature(execute_graph))
print("Scheduler types:", SequentialScheduler.__name__, ParallelScheduler.__name__)

print("All imports successful!")