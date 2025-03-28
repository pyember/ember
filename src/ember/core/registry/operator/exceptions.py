"""Custom exception definitions for Ember Operator Framework.

This module provides a compatibility layer that re-exports exceptions from the
core exceptions module while maintaining backward compatibility with existing code.
Prefer using the exceptions directly from ember.core.exceptions in new code.
"""

from ember.core.exceptions import (
    BoundMethodNotInitializedError,
    EmberError,
    FlattenError,
    OperatorError,
    OperatorExecutionError,
    OperatorSpecificationError,
    SpecificationValidationError,
    TreeTransformationError,
)

# Backwards compatibility aliases
EmberException = EmberError
OperatorSpecificationNotDefinedError = OperatorSpecificationError

# Re-export all operator exceptions for backward compatibility
__all__ = [
    "EmberException",
    "EmberError",
    "OperatorError",
    "FlattenError",
    "OperatorSpecificationNotDefinedError",
    "OperatorSpecificationError",
    "SpecificationValidationError",
    "OperatorExecutionError",
    "BoundMethodNotInitializedError",
    "TreeTransformationError",
]
