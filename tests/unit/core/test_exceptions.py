"""
Tests for the exception architecture.

This module verifies the behavior of the unified exception system.
"""

import logging

from ember.core.exceptions import (
    DataValidationError,
    EmberError,
    ErrorGroup,
    ModelNotFoundError,
    OperatorExecutionError,
    TransformError,
    ValidationError,
)


def test_base_exception_with_context():
    """Test that EmberError properly handles context fields."""
    # Create exception with context
    exc = EmberError("Test error", context={"key": "value"})

    # Verify context is stored and included in message
    assert exc.context["key"] == "value"
    assert "key='value'" in str(exc)


def test_error_code_inclusion():
    """Test that error codes are properly included."""
    # Create exception with explicit error code
    exc = EmberError("Test error", error_code=1234)

    # Verify error code is included in message
    assert "[Error 1234]" in str(exc)
    assert exc.error_code == 1234

    # Create exception with default error code
    exc = ValidationError("Invalid data")

    # Verify default error code is used
    assert "[Error 1002]" in str(exc)
    assert exc.error_code == ValidationError.DEFAULT_ERROR_CODE


def test_recovery_hint():
    """Test that recovery hints are properly included."""
    # Create exception with explicit recovery hint
    exc = EmberError("Test error", recovery_hint="Try this fix")

    # Verify recovery hint is included in message
    assert "[Recovery: Try this fix]" in str(exc)

    # Create exception with default recovery hint
    exc = ValidationError("Invalid data")

    # Verify default recovery hint is used
    assert "[Recovery:" in str(exc)


def test_exception_chaining():
    """Test exception chaining with cause."""
    # Create a cause exception
    cause = ValueError("Original error")

    # Create exception with cause
    exc = EmberError("Wrapped error", cause=cause)

    # Verify cause is stored
    assert exc.__cause__ is cause

    # Verify cause info is in context
    assert exc.context["cause_type"] == "ValueError"
    assert exc.context["cause_message"] == "Original error"


def test_error_group():
    """Test the ErrorGroup for aggregating multiple errors."""
    # Create some errors
    errors = [
        ValidationError("Error 1"),
        ValidationError("Error 2"),
    ]

    # Create error group
    group = ErrorGroup("Multiple errors occurred", errors)

    # Verify error count in context
    assert group.context["error_count"] == 2

    # Verify errors are included in message
    message = str(group)
    assert "Error 1" in message
    assert "Error 2" in message


def test_model_not_found_factory_method():
    """Test factory method for ModelNotFoundError."""
    # Create exception using factory method
    exc = ModelNotFoundError.for_model("gpt-4", provider_name="openai")

    # Verify message and context
    assert "Model 'gpt-4' not found from provider 'openai'" in str(exc)
    assert exc.context["model_name"] == "gpt-4"
    assert exc.context["provider_name"] == "openai"


def test_operator_execution_error_factory_method():
    """Test factory method for OperatorExecutionError."""
    # Create exception using factory method
    exc = OperatorExecutionError.for_operator(
        "MyOperator", cause=ValueError("Bad value")
    )

    # Verify message and context
    assert "Execution error in operator 'MyOperator'" in str(exc)
    assert "Bad value" in str(exc)
    assert exc.context["operator_name"] == "MyOperator"


def test_data_validation_error_factory_method():
    """Test factory method for DataValidationError."""
    # Create exception using factory method
    exc = DataValidationError.for_field(
        "username", expected_type="str", actual_value=123
    )

    # Verify message and context
    assert "Validation failed for field 'username'" in str(exc)
    assert "expected str" in str(exc)
    assert exc.context["field_name"] == "username"
    assert exc.context["expected_type"] == "str"
    assert "123" in exc.context["actual_value"]


def test_transform_error_factory_method():
    """Test factory method for TransformError."""
    # Create exception using factory method
    exc = TransformError.for_transform(
        "vmap",
        details={"axis": 0, "batch_size": 10},
        cause=ValueError("Inconsistent batch size"),
    )

    # Verify message and context
    assert "Error in XCS transform 'vmap'" in str(exc)
    assert "Inconsistent batch size" in str(exc)
    assert exc.context["transform_name"] == "vmap"
    assert exc.context["axis"] == 0
    assert exc.context["batch_size"] == 10


def test_exception_logging(caplog):
    """Test exception logging with context."""
    # Create logger
    logger = logging.getLogger("test")

    # Create exception
    exc = ValidationError("Test error").add_context(request_id="abc123")

    # Log exception
    with caplog.at_level(logging.ERROR):
        exc.log_with_context(logger)

    # Verify log record
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "ERROR"
    assert "ValidationError: Test error" in record.message
    assert record.structured_data["request_id"] == "abc123"


def test_method_chaining():
    """Test method chaining for add_context."""
    # Create exception with method chaining
    exc = EmberError("Test error").add_context(a=1).add_context(b=2)

    # Verify context
    assert exc.context["a"] == 1
    assert exc.context["b"] == 2
