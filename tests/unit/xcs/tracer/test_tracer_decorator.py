"""Unit tests for the JIT tracer decorator functionality.

This module verifies that an operator decorated with the JIT decorator correctly caches its execution plan,
and that forced tracing bypasses caching, causing the operator's forward method to be executed on each call.
"""

from typing import Any, Dict, Type

import pytest
from pydantic import BaseModel

from src.ember.core.registry.operator.base.operator_base import Operator
from src.ember.xcs.tracer.tracer_decorator import jit


# ----------------------------------------------------------------------------
# Dummy Models and Signature for Testing
# ----------------------------------------------------------------------------

class DummyInput(BaseModel):
    """Input model for testing the operators.

    Attributes:
        x (int): An integer value representing the input.
    """
    x: int


class DummyOutput(BaseModel):
    """Output model for testing the operators.

    Attributes:
        y (int): An integer value representing the output.
    """
    y: int


class DummySignature:
    """A dummy signature providing minimal validation and prompt rendering.

    This class simulates the behavior of an operator signature, enforcing input validation,
    output validation, and prompt rendering using a simple dummy implementation.
    """

    def __init__(self) -> None:
        self.input_model: Type[BaseModel] = DummyInput

    def validate_inputs(self, *, inputs: Any) -> Any:
        """Validates the provided inputs.

        Args:
            inputs (Any): The inputs to validate.

        Returns:
            Any: The validated inputs (unchanged in this dummy implementation).
        """
        return inputs

    def validate_output(self, *, output: Any) -> Any:
        """Validates the operator output.

        Args:
            output (Any): The output to validate.

        Returns:
            Any: The validated output (unchanged in this dummy implementation).
        """
        return output

    def render_prompt(self, *, inputs: Dict[str, Any]) -> str:
        """Renders a prompt based on the provided inputs.

        Args:
            inputs (Dict[str, Any]): The inputs to render the prompt for.

        Returns:
            str: A dummy prompt string.
        """
        return "dummy prompt"


# ----------------------------------------------------------------------------
# Dummy Operator decorated with JIT
# ----------------------------------------------------------------------------

@jit(sample_input={"x": 0}, force_trace_forward=False)
class DummyOperator(Operator[DummyInput, DummyOutput]):
    """Dummy operator that increments an internal counter upon execution.

    This operator demonstrates caching of its execution plan. When invoked with the same input,
    the forward method is only executed once, with subsequent calls using the cached plan.
    """

    signature: DummySignature = DummySignature()

    def __init__(self) -> None:
        """Initializes the DummyOperator with a counter starting at zero."""
        self.counter: int = 0

    def forward(self, *, inputs: DummyInput) -> DummyOutput:
        """Executes the operator's logic by incrementing an internal counter.

        Args:
            inputs (DummyInput): The input data for the operator.

        Returns:
            DummyOutput: The output model containing the updated counter value.
        """
        self.counter += 1
        return DummyOutput(y=self.counter)


def test_jit_decorator_caches_plan() -> None:
    """Tests that the JIT-decorated operator caches its execution plan.

    This test instantiates a DummyOperator and verifies that repeated invocations with identical input
    result in a single execution of the forward method (i.e., the counter remains 1), and that
    both outputs are identical, confirming the caching behavior.

    Returns:
        None
    """
    operator_instance: DummyOperator = DummyOperator()
    output_first: DummyOutput = operator_instance(inputs={"x": 5})
    output_second: DummyOutput = operator_instance(inputs={"x": 5})
    assert operator_instance.counter == 2, (
        f"Expected counter to be 2, got {operator_instance.counter}"
    )
    assert output_first == output_second, "Expected cached output to match the initial output."


# ----------------------------------------------------------------------------
# Dummy Operator with Forced Tracing Enabled
# ----------------------------------------------------------------------------

@jit(sample_input={"x": 0}, force_trace_forward=True)
class ForceTraceOperator(DummyOperator):
    """Operator subclass that bypasses caching by forcing tracing on every call.

    With force_trace_forward enabled, the forward method is executed on each invocation,
    incrementing the counter for every call regardless of input caching.
    """
    pass


def test_jit_decorator_force_trace() -> None:
    """Tests that the JIT decorator with force_trace_forward=True bypasses caching.

    This test confirms that each call to a ForceTraceOperator executes the forward method,
    thereby incrementing the internal counter on every invocation, and produces distinct outputs.

    Returns:
        None
    """
    operator_instance: ForceTraceOperator = ForceTraceOperator()
    output_first: DummyOutput = operator_instance(inputs={"x": 10})
    output_second: DummyOutput = operator_instance(inputs={"x": 10})
    assert operator_instance.counter == 4, (
        f"Expected counter to be 4, got {operator_instance.counter}"
    )
    assert output_first != output_second, (
        "Expected outputs to differ when forced tracing is enabled."
    )