"""
Tests for Operator initialization behavior.

This module tests that Operator subclasses don't need to call super().__init__() explicitly.
"""

import unittest
from typing import Any, Dict

from pydantic import BaseModel

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.prompt_specification.specification import Specification
from ember.core.registry.operator.base._module import ember_field


# Models defined outside of test classes to avoid collection warnings
class OpTestInput(BaseModel):
    """Simple input model for testing."""

    value: int


class OpTestOutput(BaseModel):
    """Simple output model for testing."""

    result: int


class NoSuperInitOperator(Operator[OpTestInput, OpTestOutput]):
    """Test operator that doesn't call super().__init__()."""

    specification = Specification(input_model=OpTestInput, structured_output=OpTestOutput)
    multiplier: int
    computed_field: str = ember_field(init=False)

    def __init__(self, *, multiplier: int) -> None:
        # Deliberately NOT calling super().__init__()
        self.multiplier = multiplier
        self.computed_field = f"Multiplier: {multiplier}"

    def forward(self, *, inputs: OpTestInput) -> OpTestOutput:
        """Multiply the input value by the multiplier."""
        return OpTestOutput(result=inputs.value * self.multiplier)


class WithSuperInitOperator(Operator[OpTestInput, OpTestOutput]):
    """Test operator that does call super().__init__()."""

    specification = Specification(input_model=OpTestInput, structured_output=OpTestOutput)
    multiplier: int
    computed_field: str = ember_field(init=False)

    def __init__(self, *, multiplier: int) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.computed_field = f"Multiplier: {multiplier}"

    def forward(self, *, inputs: OpTestInput) -> OpTestOutput:
        """Multiply the input value by the multiplier."""
        return OpTestOutput(result=inputs.value * self.multiplier)


class TestOperatorInitBehavior(unittest.TestCase):
    """Tests for Operator initialization behavior."""

    def test_no_super_init(self) -> None:
        """Test that an operator works without calling super().__init__()."""
        op = NoSuperInitOperator(multiplier=3)
        result = op(inputs=OpTestInput(value=5))
        self.assertEqual(result.result, 15)
        self.assertEqual(op.computed_field, "Multiplier: 3")

    def test_with_super_init(self) -> None:
        """Test that an operator works with calling super().__init__()."""
        op = WithSuperInitOperator(multiplier=3)
        result = op(inputs=OpTestInput(value=5))
        self.assertEqual(result.result, 15)
        self.assertEqual(op.computed_field, "Multiplier: 3")

    def test_dict_input(self) -> None:
        """Test that an operator works with dictionary input."""
        op = NoSuperInitOperator(multiplier=3)
        result = op(inputs={"value": 5})
        self.assertEqual(result.result, 15)

    def test_attributes(self) -> None:
        """Test that operator attributes are properly set."""
        op = NoSuperInitOperator(multiplier=3)
        self.assertEqual(op.multiplier, 3)
        self.assertEqual(op.computed_field, "Multiplier: 3")
