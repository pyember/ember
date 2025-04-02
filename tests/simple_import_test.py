"""Simplified import test after refactoring."""

import sys
import os

# Add src to path for imports
src_dir = os.path.join(os.path.dirname(__file__), '../src')
sys.path.insert(0, src_dir)

# Just test the core imports
print("Testing XCS JIT import...")
from ember.xcs.jit import jit, JITMode
print("JIT modes:", [mode.value for mode in JITMode])

print("\nTesting transforms import...")
from ember.xcs.transforms import vmap, pmap
print("vmap and pmap imported successfully")

print("\nTesting engine import...")
from ember.xcs.engine.unified_engine import (
    execute_graph, 
    ExecutionOptions,
    GraphExecutor
)
print("Engine components imported successfully")

print("\nTesting schedulers import...")
from ember.xcs.schedulers.unified_scheduler import (
    SequentialScheduler, 
    ParallelScheduler
)
print("Scheduler implementations imported successfully")

print("\nAll imports successful!")