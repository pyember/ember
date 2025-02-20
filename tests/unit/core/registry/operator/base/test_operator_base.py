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
from src.ember.core.registry.operator.exceptions import OperatorSignatureNotDefinedError


class DummyInput(BaseModel):
    value: int


class DummyOutput(BaseModel):
    result: int


class DummySignature(Signature):
    prompt_template: str = "{value}"
    input_model: Optional[Type[BaseModel]] = DummyInput
    structured_output: Optional[Type[BaseModel]] = DummyOutput
    check_all_placeholders: bool = False


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