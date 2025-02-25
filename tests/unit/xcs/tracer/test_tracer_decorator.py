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


@jit()
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


def test_jit_decorator_always_executes() -> None:
    """Tests that the JIT-decorated operator executes forward for every call (no caching)."""
    operator_instance: DummyOperator = DummyOperator()
    output_first: DummyOutput = operator_instance(inputs={"x": 5})
    output_second: DummyOutput = operator_instance(inputs={"x": 5})
    # If forward increments self.counter, we expect it to be 2 now.
    assert (
        operator_instance.counter == 2
    ), f"Expected counter to be 2, got {operator_instance.counter}"
    # The new design does NOT cache outputs, so they differ by the updated counter.
    assert (
        output_first != output_second
    ), "Expected different outputs with each call (no caching)."


# ----------------------------------------------------------------------------
# Dummy Operator with Forced Tracing Enabled
# ----------------------------------------------------------------------------


@jit(force_trace=True)
class ForceTraceOperator(DummyOperator):
    """Operator that is forced to create a trace record on every call."""

    pass


def test_jit_decorator_force_trace() -> None:
    """Tests that the JIT decorator with force_trace=True executes forward each time."""
    operator_instance: ForceTraceOperator = ForceTraceOperator()
    output_first: DummyOutput = operator_instance(inputs={"x": 10})
    output_second: DummyOutput = operator_instance(inputs={"x": 10})
    # With no caching, the counter increments on each invocation.
    assert (
        operator_instance.counter == 2
    ), f"Expected counter to be 2, but got {operator_instance.counter}"
    assert (
        output_first != output_second
    ), "Expected distinct output due to forced trace."
