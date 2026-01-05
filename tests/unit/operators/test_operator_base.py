"""Tests for the Operator base class."""

import pytest
from pydantic import ValidationError

from ember._internal.types import EmberModel
from ember.operators import Operator


class TestOperatorBase:
    """Test the Operator base class behavior."""

    def test_operator_requires_forward(self):
        """Test that Operator requires forward() implementation."""
        # Base operator without forward() should raise
        op = Operator()

        with pytest.raises(NotImplementedError) as exc_info:
            op("test")

        assert "must implement forward()" in str(exc_info.value)

    def test_simple_operator(self):
        """Test creating a simple operator subclass."""

        class DoubleOperator(Operator):
            def forward(self, x):
                return x * 2

        op = DoubleOperator()
        result = op(5)
        assert result == 10

    def test_operator_with_validation(self):
        """Test operator with input/output validation."""

        # Define validation schemas
        class InputSpec(EmberModel):
            value: int
            multiplier: int = 2

        class OutputSpec(EmberModel):
            result: int

        class ValidatedOperator(Operator):
            input_spec = InputSpec
            output_spec = OutputSpec

            def forward(self, input: InputSpec) -> OutputSpec:
                result = input.value * input.multiplier
                return OutputSpec(result=result)

        op = ValidatedOperator()

        # Dict input gets validated
        result = op({"value": 5})
        assert result.result == 10

        # Can pass validated object directly
        result = op(InputSpec(value=3, multiplier=4))
        assert result.result == 12

    def test_operator_without_validation(self):
        """Test operator without validation specs."""

        class SimpleOperator(Operator):
            def forward(self, x):
                if isinstance(x, dict):
                    return x.get("value", 0) + x.get("y", 1)
                return x + 1

        op = SimpleOperator()

        # Direct calls work
        assert op(5) == 6
        assert op({"value": 5, "y": 3}) == 8
        assert op({"value": 5, "y": 10}) == 15

    def test_operator_inheritance(self):
        """Test operator inheritance and composition."""

        class BaseProcessor(Operator):
            def preprocess(self, x):
                return x.strip().lower()

            def forward(self, x):
                return self.preprocess(x)

        class ExtendedProcessor(BaseProcessor):
            def forward(self, x):
                # Use parent preprocessing
                processed = self.preprocess(x)
                # Add own logic
                return processed.replace(" ", "_")

        op = ExtendedProcessor()
        result = op("  Hello World  ")
        assert result == "hello_world"

    def test_operator_with_state(self):
        """Test operator with internal state (immutable via equinox)."""
        import jax.numpy as jnp

        class StatefulOperator(Operator):
            count: jnp.ndarray

            def __init__(self, initial_count=0):
                self.count = jnp.array(initial_count)

            def forward(self, x):
                # Note: This doesn't mutate self.count
                # In real use, state updates happen via JAX transforms
                return x + self.count

        op = StatefulOperator(10)
        result = op(5)
        assert result == 15

        # State is immutable
        op(20)  # Doesn't change count
        assert op.count == 10

    def test_operator_error_propagation(self):
        """Test that errors in forward() propagate correctly."""

        class ErrorOperator(Operator):
            def forward(self, x):
                if x < 0:
                    raise ValueError("Negative values not allowed")
                return x * 2

        op = ErrorOperator()

        # Normal operation
        assert op(5) == 10

        # Error case
        with pytest.raises(ValueError) as exc_info:
            op(-1)
        assert "Negative values not allowed" in str(exc_info.value)

    def test_operator_supports_multiple_arguments(self):
        """Operator subclasses automatically support positional and keyword args."""

        class MathOperator(Operator):
            def forward(self, x: int, y: int, scale: int = 1, *, bias: int = 0) -> int:
                return (x + y) * scale + bias

        op = MathOperator()

        assert op(2, 3) == 5
        assert op(2, 3, scale=2) == 10
        assert op(2, 3, scale=2, bias=1) == 11
        assert op(x=1, y=4, bias=2) == 7

    def test_operator_multi_argument_validation(self):
        """input_spec applies to multi-argument operators."""

        class Inputs(EmberModel):
            x: int
            y: int
            scale: int = 1
            bias: int = 0

        class MultiValidated(Operator):
            input_spec = Inputs

            def forward(self, x: int, y: int, scale: int = 1, *, bias: int = 0) -> int:
                return (x + y) * scale + bias

        op = MultiValidated()

        assert op(2, 3) == 5
        assert op(2, 3, scale=2, bias=1) == 11

        with pytest.raises(ValidationError):
            op(2, "three")

    def test_operator_allows_disabling_cached_specs(self):
        """Explicitly setting specs to None disables validation even after caching."""

        class InputSpec(EmberModel):
            value: int

        class OutputSpec(EmberModel):
            value: int

        class ToggleOperator(Operator):
            input_spec = InputSpec
            output_spec = OutputSpec

            def forward(self, payload):
                if isinstance(payload, dict):
                    return payload
                return OutputSpec(value=payload.value)

        op = ToggleOperator()

        # Prime the caches with validation enabled
        assert op({"value": 1}).value == 1

        ToggleOperator.input_spec = None
        ToggleOperator.output_spec = None

        try:
            # Validation stays disabled even though the cache was primed
            payload = {"value": "not an int"}
            result = op(payload)
            assert result == payload
        finally:
            # Restore class attributes so other tests see the original defaults
            ToggleOperator.input_spec = InputSpec
            ToggleOperator.output_spec = OutputSpec
