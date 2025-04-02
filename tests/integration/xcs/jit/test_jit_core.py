"""Integration tests for the core JIT functionality.

Validates that JIT decoration properly optimizes Ember operators.
"""

import pytest
from typing import Dict, Any, ClassVar, List

from ember.core.registry.operator.base.operator_base import Operator, Specification
from ember.xcs import jit


class SimpleOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    """Simple operator for testing JIT compilation."""
    
    specification: ClassVar[Specification] = Specification()
    
    def __init__(self, *, value: int = 1) -> None:
        self.value = value
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = inputs.get("value", 0) + self.value
        return {"value": result}


class CompositeOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    """Composite operator containing other operators."""
    
    specification: ClassVar[Specification] = Specification()
    
    def __init__(self) -> None:
        self.op1 = SimpleOperator(value=5)
        self.op2 = SimpleOperator(value=10)
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Chain operations
        intermediate = self.op1(inputs=inputs)
        return self.op2(inputs=intermediate)


def test_jit_basic_operator():
    """Test JIT compilation of a basic operator."""
    # Create and JIT-compile the operator
    op = SimpleOperator(value=42)
    jit_op = jit(op)
    
    # Execute the operator
    result = jit_op(inputs={"value": 10})
    
    # Verify the result
    assert "value" in result
    assert result["value"] == 52  # 10 + 42


def test_jit_composite_operator():
    """Test JIT compilation of a composite operator."""
    # Create and JIT-compile the operator
    op = CompositeOperator()
    jit_op = jit(op)
    
    # Execute the operator
    result = jit_op(inputs={"value": 0})
    
    # Verify the result
    assert "value" in result
    assert result["value"] == 15  # 0 + 5 + 10


@jit
class DecoratedOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    """Class decorated with JIT."""
    
    specification: ClassVar[Specification] = Specification()
    
    def __init__(self, *, value: int = 1) -> None:
        self.value = value
    
    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = inputs.get("value", 0) * self.value
        return {"value": result}


def test_jit_class_decoration():
    """Test JIT decoration directly on a class."""
    # Create an instance of the decorated class
    op = DecoratedOperator(value=5)
    
    # Execute the operator (already JIT-compiled)
    result = op(inputs={"value": 10})
    
    # Verify the result
    assert "value" in result
    assert result["value"] == 50  # 10 * 5