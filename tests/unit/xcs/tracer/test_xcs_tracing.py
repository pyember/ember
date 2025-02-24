"""Unit tests for TracerContext functionality.

This module tests that TracerContext correctly patches operators, builds an IRGraph,
captures trace records, and restores the original operator methods.
"""

from typing import Any, Dict
import pytest
from pydantic import BaseModel

from src.ember.core.registry.operator.base.operator_base import Operator
from src.ember.xcs.tracer.xcs_tracing import TracerContext
from src.ember.xcs.tracer.tracer_decorator import jit


class MockInput(BaseModel):
    """Input model for the mock operator."""

    value: int


@jit()
class MockOperator(Operator[MockInput, Dict[str, Any]]):
    """A mock operator that doubles the input value."""

    # For testing, we use a simplified signature.
    signature = type(
        "DummySignature",
        (),
        {
            "input_model": MockInput,
            "validate_inputs": lambda self, *, inputs: inputs,
            "validate_output": lambda self, *, output: output,
            "render_prompt": lambda self, *, inputs: "dummy prompt",
        },
    )()

    def forward(self, *, inputs: Any) -> Dict[str, Any]:
        # Allow inputs to be passed as either a dict or a MockInput instance.
        if isinstance(inputs, dict):
            inputs = MockInput(**inputs)
        return {"result": inputs.value * 2}


def test_tracer_context_basic() -> None:
    """Tests basic tracing with TracerContext."""
    operator = MockOperator()
    sample_input = {"value": 5}
    with TracerContext() as tracer:
        _ = operator(inputs=sample_input)
    assert len(tracer.records) >= 1, "Expected at least one trace record."
    first_record = tracer.records[0]
    assert first_record.outputs == {"result": 10}, (
        f"Traced output {first_record.outputs} does not match expected {{'result': 10}}."
    )


def test_tracer_context_patch_restore() -> None:
    """Tests that operator patching is no longer performed (or is preserved) in the new design."""
    operator = MockOperator()
    original_call = operator.__class__.__call__
    with TracerContext() as _:
        pass
    assert (
        operator.__class__.__call__ == original_call
    ), "Operator __call__ should remain unchanged in the new implementation."
