"""
Tests for the unified JIT interface.

This module verifies that the unified JIT interface correctly dispatches
to the appropriate specialized implementation based on the selected mode.
"""

import pytest

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.types.ember_model import EmberModel
from ember.xcs.tracer.unified_jit import jit


class TestInput(EmberModel):
    """Simple input model for testing."""

    value: str


class TestOutput(EmberModel):
    """Simple output model for testing."""

    result: str


class SimpleOperator(Operator[TestInput, TestOutput]):
    """A simple operator for testing JIT implementations."""

    from typing import ClassVar

    from ember.core.registry.specification.specification import Specification

    specification: ClassVar[Specification] = Specification()

    def forward(self, *, inputs: TestInput) -> TestOutput:
        """Process the input and return a result."""
        return TestOutput(result=f"processed_{inputs.value}")


def test_unified_jit_default_mode():
    """Test that the default mode (trace) works correctly."""

    @jit
    class TestOp(SimpleOperator):
        pass

    op = TestOp()
    result = op(inputs=TestInput(value="test"))

    assert result.result == "processed_test"


def test_unified_jit_trace_mode():
    """Test explicitly selecting trace mode."""

    @jit(mode="trace")
    class TestOp(SimpleOperator):
        pass

    op = TestOp()
    result = op(inputs=TestInput(value="test"))

    assert result.result == "processed_test"


def test_unified_jit_structural_mode():
    """Test selecting structural mode."""

    @jit(mode="structural")
    class TestOp(SimpleOperator):
        pass

    op = TestOp()
    result = op(inputs=TestInput(value="test"))

    assert result.result == "processed_test"
    # Verify this is a structural JIT by checking if it has a _jit_enabled attribute
    # (this is implementation-specific but useful for testing)
    assert hasattr(op, "_jit_enabled") or hasattr(type(op), "_jit_enabled")


def test_unified_jit_trace_explicit():
    """Test using the explicit trace interface."""

    # Create a proper sample input
    sample = TestInput(value="sample")

    @jit.trace(sample_input=sample)
    class TestOp(SimpleOperator):
        pass

    op = TestOp()
    result = op(inputs=TestInput(value="test"))

    assert result.result == "processed_test"


def test_unified_jit_structural_explicit():
    """Test using the explicit structural interface."""

    @jit.structural(execution_strategy="sequential")
    class TestOp(SimpleOperator):
        pass

    op = TestOp()
    result = op(inputs=TestInput(value="test"))

    assert result.result == "processed_test"
    # Verify this is a structural JIT by checking if it has a _jit_enabled attribute
    assert hasattr(op, "_jit_enabled") or hasattr(type(op), "_jit_enabled")


def test_unified_jit_invalid_mode():
    """Test that an invalid mode raises a ValueError."""

    with pytest.raises(ValueError):

        @jit(mode="invalid")
        class TestOp(SimpleOperator):
            pass
