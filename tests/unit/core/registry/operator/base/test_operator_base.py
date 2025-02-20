"""
Tests for the core Operator base functionality.

These tests cover:
  - Execution of a dummy operator.
  - Registration of sub-operators.
  - Building inputs when an input model is defined.
  - Correct error handling when signatures are missing or misconfigured.
"""

from typing import Any, Dict, Optional, Type
import pytest
from pydantic import BaseModel
from src.ember.core.registry.operator.base.operator_base import Operator
from src.ember.core.registry.prompt_signature.signatures import Signature
from src.ember.core.registry.operator.exceptions import (
    OperatorSignatureNotDefinedError,
    SignatureValidationError,
    OperatorExecutionError,
)


class DummyInput(BaseModel):
    value: int


class DummyOutput(BaseModel):
    result: int


class DummySignature(Signature):
    prompt_template: str = "{value}"
    input_model: Optional[Type[BaseModel]] = DummyInput
    structured_output: Optional[Type[BaseModel]] = DummyOutput
    check_all_placeholders: bool = False

    def validate_inputs(self, inputs: Any) -> DummyInput:
        return DummyInput(**inputs)

    def validate_output(self, output: Any) -> DummyOutput:
        return DummyOutput(
            **(output.model_dump() if hasattr(output, "model_dump") else output)
        )


class AddOneOperator(Operator[DummyInput, DummyOutput]):
    signature: Signature = DummySignature()

    def forward(self, *, inputs: DummyInput) -> DummyOutput:
        return DummyOutput(result=inputs.value + 1)


def test_operator_call_valid() -> None:
    op = AddOneOperator()
    inputs: Dict[str, int] = {"value": 10}
    output = op(inputs=inputs)
    assert output.result == 11


def test_missing_signature_error() -> None:
    """Test that calling an operator without a signature raises OperatorSignatureNotDefinedError."""

    class NoSignatureOperator(Operator):
        signature = None

        def forward(self, *, inputs: Any) -> Any:
            return inputs

    operator = NoSignatureOperator()
    with pytest.raises(OperatorSignatureNotDefinedError) as exc_info:
        operator(inputs={"value": "test"})
    assert str(exc_info.value) == "Operator signature must be defined."


def test_operator_forward_exception() -> None:
    class FailingOperator(Operator[DummyInput, DummyOutput]):
        signature: Signature = DummySignature()

        def forward(self, *, inputs: DummyInput) -> DummyOutput:
            raise ValueError("Intentional failure")

    op = FailingOperator()
    with pytest.raises(OperatorExecutionError):
        op(inputs={"value": 5})


def test_output_validation_failure() -> None:
    class InvalidOutputOperator(Operator[DummyInput, DummyOutput]):
        signature: Signature = DummySignature()

        def forward(self, *, inputs: DummyInput) -> DummyOutput:
            # Return a dict instead of DummyOutput to trigger validation in __call__
            return {"result": "not an int"}

    op = InvalidOutputOperator()
    with pytest.raises(SignatureValidationError):
        op(inputs={"value": 5})
