from __future__ import annotations
from typing import Optional


class PromptSpecificationError(Exception):
    """
    Base class for all prompt specification related errors.

    Attributes:
        message (str): Description of the error.
        error_code (Optional[int]): An optional error code representing the error type.
    """

    def __init__(self, message: str, error_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.error_code: Optional[int] = error_code

    def __str__(self) -> str:
        return (
            f"[Error {self.error_code}] {self.message}"
            if self.error_code is not None
            else self.message
        )


class PlaceholderMissingError(PromptSpecificationError):
    """
    Raised when a required placeholder is missing in the prompt template.

    Attributes:
        missing_placeholder (Optional[str]): The name(s) of the missing placeholder(s).
    """

    DEFAULT_ERROR_CODE: int = 1001

    def __init__(self, message: str, missing_placeholder: Optional[str] = None) -> None:
        if missing_placeholder and missing_placeholder not in message:
            message = f"Missing placeholder(s) '{missing_placeholder}': {message}"
        super().__init__(message, error_code=self.DEFAULT_ERROR_CODE)
        self.missing_placeholder: Optional[str] = missing_placeholder


class MismatchedModelError(PromptSpecificationError):
    """
    Raised when a provided Pydantic model instance does not match the expected type.
    """

    DEFAULT_ERROR_CODE: int = 1002

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code=self.DEFAULT_ERROR_CODE)


class InvalidInputTypeError(PromptSpecificationError):
    """
    Raised when the input or output data type is not a dict or a Pydantic model.
    """

    DEFAULT_ERROR_CODE: int = 1003

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code=self.DEFAULT_ERROR_CODE)