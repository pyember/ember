"""Custom exception definitions for Ember.

These exceptions provide granular error handling for signature validation,
operator execution, and initialization, making debugging and client error‐handling
more explicit and robust.
"""


class EmberException(Exception):
    """Base exception class for all Ember errors."""

    pass


class OperatorSignatureNotDefinedError(EmberException):
    """Raised when an Operator's signature is not defined."""

    def __init__(self, message: str = "Operator signature must be defined."):
        super().__init__(message)


class SignatureValidationError(EmberException):
    """Raised when input or output signature validation fails."""

    def __init__(self, message: str = "Signature validation error occurred."):
        super().__init__(message)


class OperatorExecutionError(EmberException):
    """Raised when an error occurs during operator execution."""

    def __init__(
        self, message: str = "An error occurred during operator execution."
    ) -> None:
        """Initialize the error with an optional custom message.

        Args:
            message: The error message to display.
        """
        super().__init__(message)


class BoundMethodNotInitializedError(EmberException):
    """Raised when a BoundMethod is not properly initialized with its function and self."""

    def __init__(
        self,
        message: str = "BoundMethod not properly initialized with __func__ and __self__.",
    ):
        super().__init__(message)


class TreeTransformationError(Exception):
    pass
