"""Unified JIT compilation interface for Ember operators.

Provides access to both trace-based and structure-based JIT strategies through
a single decorator. Handles mode selection and dispatches to specialized
implementations.

Example:
    Default trace-based JIT:
    ```python
    @jit
    class SimpleOperator(Operator):
        def forward(self, *, inputs):
            return process(inputs)
    ```
    
    JIT with explicit strategy:
    ```python
    @jit(mode="structural")
    class StructuredOperator(Operator):
        def forward(self, *, inputs):
            return process(inputs)
    ```
    
    Strategy-specific configuration:
    ```python
    @jit.trace(sample_input=sample_data)
    class TracedOperator(Operator):
        def forward(self, *, inputs):
            return complex_process(inputs)
            
    @jit.structural(execution_strategy="parallel")
    class ParallelOperator(Operator):
        def __init__(self):
            self.op1 = SubOperator()
            self.op2 = SubOperator()
            
        def forward(self, *, inputs):
            result1 = self.op1(inputs=inputs)
            result2 = self.op2(inputs=inputs)
            return combine(result1, result2)
    ```
"""

from __future__ import annotations

from typing import Callable, Type, TypeVar, overload

from ember.xcs.tracer.structural_jit import structural_jit

# Import specialized JIT implementations
from ember.xcs.tracer.tracer_decorator import jit as trace_jit

# Type variable for operator classes
T = TypeVar("T")


@overload
def jit(func: Type[T]) -> Type[T]:
    ...


@overload
def jit(*, mode: str = "trace") -> Callable[[Type[T]], Type[T]]:
    ...


def jit(func=None, *, mode: str = "trace"):
    """Just-in-time compiler for operator optimization.

    Transforms operator classes for automatic graph-based execution.
    Selects between tracing and structural analysis strategies.

    Args:
        func: Operator class to decorate
        mode: JIT strategy:
            - "trace": Records and optimizes execution paths
            - "structural": Analyzes composition structure for parallelism

    Returns:
        Decorated operator class

    Raises:
        ValueError: If unknown mode specified
    """
    MODES = {"trace": trace_jit, "structural": structural_jit}

    def decorator(cls: Type[T]) -> Type[T]:
        if mode not in MODES:
            raise ValueError(
                f"Unknown JIT mode: {mode}. Valid options: {', '.join(MODES.keys())}"
            )
        return MODES[mode](cls)

    # Handle both @jit and @jit(...) syntax
    return decorator(func) if func is not None else decorator


# Direct access to specialized JIT implementations
jit.trace = trace_jit
jit.structural = structural_jit

# Legacy aliases for backward compatibility
trace_based_jit = trace_jit
structure_based_jit = structural_jit
