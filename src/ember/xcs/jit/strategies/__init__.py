"""JIT compilation strategies.

Provides various strategies for JIT compilation, including trace-based,
structural, and enhanced approaches. Each strategy has strengths for
different operator patterns.
"""

from ember.xcs.jit.strategies.base_strategy import Strategy, JITFallbackMixin
from ember.xcs.jit.strategies.trace import TraceStrategy
from ember.xcs.jit.strategies.structural import StructuralStrategy
from ember.xcs.jit.strategies.enhanced import EnhancedStrategy

__all__ = [
    # Base protocol and utilities
    "Strategy",
    "JITFallbackMixin",
    
    # Concrete strategy implementations
    "TraceStrategy",
    "StructuralStrategy",
    "EnhancedStrategy",
]