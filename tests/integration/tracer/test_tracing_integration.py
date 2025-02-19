"""
Tests for XCS tracing and the JIT decorator.

These tests verify:
  - That TracerContext captures operator calls correctly.
  - That the traced graph can be converted to an executable plan.
  - That JIT caching prevents re-tracing and that force_trace_forward forces a new trace.
  - Re-entrancy is properly handled (including nested JIT operators).
"""

import pytest
from src.ember.core.non import Operator
from src.ember.xcs.tracer.xcs_tracing import TracerContext, convert_traced_graph_to_plan
from src.ember.xcs.tracer.tracer_decorator import jit


class DummySignature:
    def validate_inputs(self, inputs):
        return inputs

    def validate_output(self, output):
        return output


class DummyTracingOperator(Operator):
    def __init__(self):
        self.signature = DummySignature()

    def get_signature(self):
        return self.signature

    def forward(self, *, inputs: dict) -> dict:
        return {"output": "traced"}


def test_tracer_context_capture() -> None:
    op = DummyTracingOperator()
    op.name = "DummyTrace"
    with TracerContext(top_operator=op, sample_input={"dummy": "data"}) as tctx:
        traced_graph = tctx.run_trace()
    # Ensure that at least one node is captured and that our operator produced the expected output.
    assert len(traced_graph.nodes) >= 1
    found = False
    for node_id, node in traced_graph.nodes.items():
        if node.operator == op:
            assert node.attrs.get("output_0") == "traced"
            found = True
            break
    assert found, "The operator's execution was not traced."


def test_convert_traced_graph_to_plan() -> None:
    op = DummyTracingOperator()
    op.name = "DummyTracePlan"
    with TracerContext(top_operator=op, sample_input={"dummy": "data"}) as tctx:
        traced_graph = tctx.run_trace()
    plan = convert_traced_graph_to_plan(tracer_graph=traced_graph)
    assert len(plan.tasks) == len(traced_graph.nodes)


def test_jit_caching() -> None:
    @jit()
    class DummyJITOperator(Operator):
        def __init__(self):
            self.signature = DummySignature()

        def get_signature(self):
            return self.signature

        def forward(self, *, inputs: dict) -> dict:
            # Mimic some transformation by appending "_processed"
            return {"value": inputs.get("value", "default") + "_processed"}

    op = DummyJITOperator()
    op.name = "JITOp"

    output1 = op(inputs={"value": "input"})
    assert output1["value"] == "input_processed"

    # The plan compiled from the first call
    cached_plan = op._compiled_plans["default"]

    # Second invocation should reuse the same plan
    output2 = op(inputs={"value": "another"})
    assert output2["value"] == "another_processed"
    assert op._compiled_plans["default"] is cached_plan


def test_force_trace_forward() -> None:
    @jit(sample_input={"value": 1}, force_trace_forward=True)
    class DummyForceJITOperator(Operator):
        trace_count = 0

        def __init__(self):
            self.signature = DummySignature()

        def get_signature(self):
            return self.signature

        def forward(self, *, inputs: dict) -> dict:
            DummyForceJITOperator.trace_count += 1
            return {"value": inputs.get("value", 0) + 1}

    op = DummyForceJITOperator()
    op.name = "ForceJITOp"
    output1 = op(inputs={"value": 5})
    count1 = DummyForceJITOperator.trace_count
    output2 = op(inputs={"value": 5})
    count2 = DummyForceJITOperator.trace_count
    # When force_trace_forward is enabled, the trace should happen on every call
    assert count2 > count1
    assert output1["value"] == 6
    assert output2["value"] == 6


def test_nested_jit() -> None:
    @jit()
    class InnerOperator(Operator):
        def __init__(self):
            self.signature = DummySignature()

        def get_signature(self):
            return self.signature

        def forward(self, *, inputs: dict) -> dict:
            return {"inner": inputs["value"] + "_inner"}

    @jit()
    class OuterOperator(Operator):
        def __init__(self):
            self.signature = DummySignature()
            self.inner = InnerOperator()

        def get_signature(self):
            return self.signature

        def forward(self, *, inputs: dict) -> dict:
            inner_out = self.inner(inputs=inputs)
            return {"outer": inner_out["inner"] + "_outer"}

    op = OuterOperator()
    result = op(inputs={"value": "test"})
    assert result["outer"] == "test_inner_outer"

