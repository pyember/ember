"""Unit tests for TracerContext functionality.

This module tests that TracerContext correctly patches operators, builds an IRGraph,
captures trace records, and restores the original operator methods.
"""

from typing import Any, Dict
import pytest
from pydantic import BaseModel

from src.ember.core.registry.operator.base.operator_base import Operator
from src.ember.xcs.tracer.xcs_tracing import TracerContext
from src.ember.xcs.tracer.tracer_decorator import IRGraph


class MockInput(BaseModel):
    """Input model for the mock operator."""
    value: int


class MockOperator(Operator[MockInput, Dict[str, Any]]):
    """A mock operator that doubles the input value."""
    # For testing, we use a simplified signature.
    signature = type("DummySignature", (), {
        "input_model": MockInput,
        "validate_inputs": lambda self, *, inputs: inputs,
        "validate_output": lambda self, *, output: output,
        "render_prompt": lambda self, *, inputs: "dummy prompt"
    })()

    def forward(self, *, inputs: Any) -> Dict[str, Any]:
        # Allow inputs to be passed as either a dict or a MockInput instance.
        if isinstance(inputs, dict):
            inputs = MockInput(**inputs)
        return {"result": inputs.value * 2}


def test_tracer_context_basic() -> None:
    """Tests basic tracing with TracerContext."""
    operator = MockOperator()
    sample_input = {"value": 5}
    with TracerContext(top_operator=operator, sample_input=sample_input) as tracer:
        ir_graph = tracer.run_trace()
    # Check that the IRGraph has at least one node.
    assert isinstance(ir_graph, IRGraph)
    assert len(ir_graph.nodes) >= 1
    # If trace records are being collected, check that at least one exists.
    if hasattr(tracer, "trace_records"):
        assert len(tracer.trace_records) >= 1
        # Verify that the traced output is correct.
        record = tracer.trace_records[0]
        assert record.outputs == {"result": 10}


def test_tracer_context_patch_restore() -> None:
    """Tests that operator patching is restored after tracing."""
    operator = MockOperator()
    original_call = operator.__class__.__call__
    with TracerContext(top_operator=operator, sample_input={"value": 1}):
        pass
    assert operator.__class__.__call__ == original_call