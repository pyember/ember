"""Exception definitions for specification module.

This module provides a compatibility layer that re-exports exceptions from the
core exceptions module while maintaining backward compatibility with existing code.
Prefer using the exceptions directly from ember.core.exceptions in new code.
"""

from ember.core.exceptions import (
    SpecificationValidationError,
    InvalidPromptError as PromptSpecificationError,
)

# Re-export specification exceptions for backward compatibility
__all__ = [
    "PromptSpecificationError",
    "SpecificationValidationError",
]
