"""Custom exception definitions for Ember.

These exceptions provide granular error handling for specification validation,
operator execution, and initialization, making debugging and client error‐handling
more explicit and robust.
"""

from typing import Optional


class EmberException(Exception):
    """Base exception class for all Ember errors."""

    pass


class OperatorError(EmberException):
    """Base class for all operator errors with optional error_code."""

    def __init__(self, message: str, error_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.error_code: Optional[int] = error_code

    def __str__(self) -> str:
        # You can also use self.args[0], but referencing super() helps in some cases
        return (
            f"[Error {self.error_code}] {super().__str__()}"
            if self.error_code is not None
            else super().__str__()
        )


class FlattenError(OperatorError):
    """Raised when flattening an EmberModule fails due to inconsistent field states."""

    DEFAULT_ERROR_CODE = 2001

    def __init__(
        self, message: str = "Failed to flatten EmberModule due to field inconsistency."
    ):
        super().__init__(message, error_code=self.DEFAULT_ERROR_CODE)


class OperatorSpecificationNotDefinedError(OperatorError):
    """Raised when an Operator's specification is not defined."""

    DEFAULT_ERROR_CODE = 2002

    def __init__(self, message: str = "Operator specification must be defined."):
        super().__init__(message, error_code=self.DEFAULT_ERROR_CODE)
        
class OperatorSpecificationNotDefinedError(OperatorSpecificationNotDefinedError):
    """Legacy exception for backward compatibility. 
    
    This is an alias for OperatorSpecificationNotDefinedError.
    """
    pass


class SpecificationValidationError(OperatorError):
    """Raised when input or output specification validation fails."""

    DEFAULT_ERROR_CODE = 2003

    def __init__(self, message: str = "Specification validation error occurred."):
        super().__init__(message, error_code=self.DEFAULT_ERROR_CODE)
        
class SpecificationValidationError(SpecificationValidationError):
    """Legacy exception for backward compatibility.
    
    This is an alias for SpecificationValidationError.
    """
    pass


class OperatorExecutionError(EmberException):
    """Raised when an error occurs during operator execution."""

    DEFAULT_ERROR_CODE = 2004

    def __init__(
        self, message: str = "An error occurred during operator execution."
    ) -> None:
        super().__init__(message)
        self.error_code: Optional[int] = self.DEFAULT_ERROR_CODE

    def __str__(self) -> str:
        return (
            f"[Error {self.error_code}] {super().__str__()}"
            if self.error_code is not None
            else super().__str__()
        )


class BoundMethodNotInitializedError(EmberException):
    """Raised when a BoundMethod is not properly initialized with its function and self."""

    DEFAULT_ERROR_CODE = 2005

    def __init__(
        self,
        message: str = "BoundMethod not properly initialized with __func__ and __self__.",
    ):
        super().__init__(message)
        self.error_code: Optional[int] = self.DEFAULT_ERROR_CODE

    def __str__(self) -> str:
        return (
            f"[Error {self.error_code}] {super().__str__()}"
            if self.error_code is not None
            else super().__str__()
        )


class TreeTransformationError(EmberException):
    """Raised when an error occurs during tree transformation."""

    DEFAULT_ERROR_CODE = 2006

    def __init__(self, message: str = "Tree transformation error occurred."):
        super().__init__(message)
        self.error_code: Optional[int] = self.DEFAULT_ERROR_CODE

    def __str__(self) -> str:
        return (
            f"[Error {self.error_code}] {super().__str__()}"
            if self.error_code is not None
            else super().__str__()
        )
