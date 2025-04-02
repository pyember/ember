"""JIT compilation modes.

Defines the set of available JIT compilation modes used throughout the system.
This module is separate to avoid circular imports between jit modules.
"""

from enum import Enum


class JITMode(str, Enum):
    """JIT compilation modes available in the system."""
    
    AUTO = "auto"  # Automatically select the best strategy
    TRACE = "trace"  # Traditional execution tracing
    STRUCTURAL = "structural"  # Structure-based analysis
    ENHANCED = "enhanced"  # Enhanced JIT with improved parallelism detection