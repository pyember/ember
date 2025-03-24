"""Exception definitions for specification module.

This module provides a compatibility layer that re-exports exceptions from the
core exceptions module while maintaining backward compatibility with existing code.
Prefer using the exceptions directly from ember.core.exceptions in new code.
"""

from ember.core.exceptions import (
    EmberError,
    InvalidArgumentError,
    SpecificationValidationError,
)


# Legacy exceptions for backward compatibility
class PromptSpecificationError(SpecificationValidationError):
    """Legacy base class for prompt specification errors.

    This class is maintained for backward compatibility.
    For new code, use SpecificationValidationError from ember.core.exceptions.
    """

    pass


class PlaceholderMissingError(PromptSpecificationError):
    """Raised when a required placeholder is missing in the prompt template.

    This class uses the SpecificationValidationError from core.exceptions internally,
    while maintaining the same interface for backward compatibility.

    Attributes:
        missing_placeholder (str): The name of the missing placeholder.
    """

    def __init__(self, message: str, missing_placeholder: str = None) -> None:
        if missing_placeholder and missing_placeholder not in message:
            message = f"Missing placeholder(s) '{missing_placeholder}': {message}"
        super().__init__(message)
        self.missing_placeholder = missing_placeholder
        self.add_context(missing_placeholder=missing_placeholder)


class MismatchedModelError(PromptSpecificationError):
    """Raised when a provided Pydantic model instance does not match the expected type."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.add_context(error_type="model_mismatch")


class InvalidInputTypeError(PromptSpecificationError):
    """Raised when the input or output data type is not a dict or a Pydantic model."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.add_context(error_type="invalid_input_type")


# Re-export all specification exceptions for backward compatibility
__all__ = [
    "PromptSpecificationError",
    "PlaceholderMissingError",
    "MismatchedModelError",
    "InvalidInputTypeError",
]
