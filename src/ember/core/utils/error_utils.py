import logging
from typing import Any, Callable, Dict, Optional, Type
from src.ember.core.registry.operator.exceptions import (
    OperatorExecutionError,
    SignatureValidationError,
)

logger = logging.getLogger(__name__)

# Error code constants.
FORWARD_ERROR_CODE: str = "E_FORWARD_COMPUTATION"
OUTPUT_VALIDATION_ERROR_CODE: str = "E_OUTPUT_VALIDATION"


def _log_and_wrap_error(
    *,
    operator: str,
    error_code: str,
    context_message: str,
    error: Exception,
    error_label: str,
    logger_method: Callable[..., None],
    exception_cls: Type[Exception],
    extra: Optional[Dict[str, Any]] = None,
) -> Exception:
    """Log an error with a formatted message and wrap it in an exception.

    Constructs a detailed error message by combining the operator identifier, error
    code, contextual message, and the original exception. The message is logged using
    the provided logging method, and an instance of the specified exception class is
    returned.

    Args:
        operator (str): Identifier of the operator.
        error_code (str): Specific error code representing the error type.
        context_message (str): Contextual information about the error.
        error (Exception): The original exception to be wrapped.
        error_label (str): Label to prefix the original error message.
        logger_method (Callable[..., None]): The logging method to invoke (e.g.,
            logger.error or logger.exception).
        exception_cls (Type[Exception]): The exception class used for wrapping the error.
        extra (Optional[Dict[str, Any]]): Additional context to be included in the log.

    Returns:
        Exception: An instance of exception_cls containing the full error message.
    """
    full_message: str = (
        f"[{operator}] [{error_code}] {context_message} | {error_label}: {error}"
    )
    logger_method(msg=full_message, extra=extra, exc_info=True)
    return exception_cls(full_message)


def wrap_forward_error(
    *,
    operator_class: str,
    validated_inputs: Any,
    error: Exception,
) -> OperatorExecutionError:
    """Wrap an error from a forward computation in an OperatorExecutionError.

    Logs detailed information about the error encountered during the forward pass,
    including the operator class and the validated inputs, then returns an
    OperatorExecutionError encapsulating these details.

    Args:
        operator_class (str): Identifier of the operator class.
        validated_inputs (Any): The inputs validated for the forward computation.
        error (Exception): The original exception encountered during the forward pass.

    Returns:
        OperatorExecutionError: An exception instance with detailed error context.
    """
    context_message: str = (
        f"Error during forward() with validated inputs: {validated_inputs}"
    )
    return _log_and_wrap_error(
        operator=operator_class,
        error_code=FORWARD_ERROR_CODE,
        context_message=context_message,
        error=error,
        error_label="Original error",
        logger_method=logger.exception,
        exception_cls=OperatorExecutionError,
    )


def wrap_validation_error(
    *,
    operator_class: str,
    operator_output: Any,
    error: Exception,
) -> SignatureValidationError:
    """Wrap an error from output validation in a SignatureValidationError.

    Logs detailed information about the output validation failure, including the
    operator class and the operator output, then returns a SignatureValidationError
    encapsulating this information.

    Args:
        operator_class (str): Identifier of the operator.
        operator_output (Any): The output produced by the operator that failed validation.
        error (Exception): The original exception encountered during output validation.

    Returns:
        SignatureValidationError: An exception instance with detailed error context.
    """
    context_message: str = (
        f"Output validation failed for operator output: {operator_output}"
    )
    extra_context: Dict[str, Any] = {
        "operator": operator_class,
        "error_code": OUTPUT_VALIDATION_ERROR_CODE,
    }
    return _log_and_wrap_error(
        operator=operator_class,
        error_code=OUTPUT_VALIDATION_ERROR_CODE,
        context_message=context_message,
        error=error,
        error_label="Error",
        logger_method=logger.error,
        exception_cls=SignatureValidationError,
        extra=extra_context,
    )