"""Tests for the @op decorator."""

from dataclasses import FrozenInstanceError
from typing import Dict, List

import jax
import jax.numpy as jnp
import pytest
from pydantic import ValidationError

from ember._internal.types import EmberModel
from ember.api.decorators import op
from ember.api.operators import Operator, chain, ensemble
from ember.xcs import jit


class TestOpDecorator:
    """Test the @op decorator functionality."""

    def test_simple_function_to_operator(self):
        """Test that @op converts a function to an operator."""

        @op
        def double(x: int) -> int:
            """Double the input."""
            return x * 2

        assert isinstance(double, Operator)

        result = double(5)
        assert result == 10

        # Materialized operator retains function metadata
        assert double.__wrapped__.__name__ == "double"

    def test_function_with_multiple_args(self):
        """Test @op with functions taking multiple arguments."""

        @op
        def add_and_multiply(x: float, y: float, multiplier: int = 2) -> float:
            return (x + y) * multiplier

        assert isinstance(add_and_multiply, Operator)
        assert add_and_multiply(2, 3, multiplier=3) == 15

    def test_composition_with_decorated_functions(self):
        """Test that @op functions work with composition."""

        @op
        def normalize(text: str) -> str:
            return text.lower().strip()

        @op
        def tokenize(text: str) -> List[str]:
            return text.split()

        @op
        def count_words(tokens: List[str]) -> int:
            return len(tokens)

        # Compose with chain
        pipeline = chain(normalize, tokenize, count_words)

        result = pipeline("  Hello WORLD  ")
        assert result == 2

    def test_ensemble_with_decorated_functions(self):
        """Test that @op functions work in ensembles."""

        @op
        def method1(x: int) -> str:
            return "positive" if x > 0 else "negative"

        @op
        def method2(x: int) -> str:
            return "even" if x % 2 == 0 else "odd"

        @op
        def method3(x: int) -> str:
            return "small" if abs(x) < 10 else "large"

        # Create ensemble
        analyzer = ensemble(method1, method2, method3)

        results = analyzer(5)
        assert results == ["positive", "odd", "small"]

    def test_jax_integration_with_decorated_function(self):
        """Test that @op functions work with JAX transformations."""

        @op
        def compute(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(x**2)

        # Should work with jax.grad
        grad_fn = jax.grad(lambda x: compute(x))
        x = jnp.array([1.0, 2.0, 3.0])
        grads = grad_fn(x)

        # Gradient of sum(x^2) is 2x
        expected = 2 * x
        assert jnp.allclose(grads, expected)

    def test_xcs_jit_with_decorated_function(self):
        """Test that @op functions can be used with XCS @jit."""

        @op
        def compute_sum(x: int) -> int:
            # Simple computation
            return sum(range(x))

        # The decorated function is an Operator
        assert isinstance(compute_sum, Operator)

        # Can apply JIT to the operator
        fast_compute = jit(compute_sum)
        assert fast_compute(10) == 45

        # Basic functionality test
        result = compute_sum(10)
        assert result == 45  # sum(0..9)

    def test_decorator_kwargs_enable_validation(self):
        class Inputs(EmberModel):
            x: int
            y: int

        @op(input_spec=Inputs)
        def add(x: int, y: int) -> int:
            return x + y

        assert isinstance(add, Operator)
        assert add(1, 2) == 3
        with pytest.raises(ValidationError):
            add(1, "two")

    def test_annotation_driven_validation(self):
        class Payload(EmberModel):
            text: str

        @op
        def shout(input: Payload) -> str:
            return input.text.upper()

        assert shout({"text": "hi"}) == "HI"
        with pytest.raises(ValidationError):
            shout({"wrong": "field"})

    def test_post_configuration_before_call(self):
        class Inputs(EmberModel):
            value: int

        @op
        def square(payload: Inputs) -> int:
            return payload.value * payload.value

        square.input_spec = Inputs
        assert square({"value": 4}) == 16
        with pytest.raises(ValidationError):
            square({"value": "oops"})

    def test_specs_locked_after_materialization(self):
        class Inputs(EmberModel):
            value: int

        @op
        def inc(value: int) -> int:
            return value + 1

        inc(1)  # materialize
        with pytest.raises(FrozenInstanceError):
            inc.input_spec = Inputs

    def test_nested_decorated_functions(self):
        """Test nested calls between decorated functions."""

        @op
        def inner(x: int) -> int:
            return x + 1

        @op
        def middle(x: int) -> int:
            return inner(x) * 2

        @op
        def outer(x: int) -> int:
            return middle(x) + inner(x)

        # outer(5) = middle(5) + inner(5)
        #          = (inner(5) * 2) + inner(5)
        #          = (6 * 2) + 6
        #          = 12 + 6 = 18
        result = outer(5)
        assert result == 18

    def test_decorated_function_in_operator_class(self):
        """Test using @op functions inside operator classes."""

        @op
        def preprocess(text: str) -> str:
            return text.strip().lower()

        class AnalysisOperator(Operator):
            def forward(self, text: str) -> Dict[str, any]:
                # Use decorated function
                cleaned = preprocess(text)
                return {"original": text, "cleaned": cleaned, "length": len(cleaned)}

        analyzer = AnalysisOperator()
        result = analyzer("  HELLO WORLD  ")

        assert result["cleaned"] == "hello world"
        assert result["length"] == 11

    def test_progressive_disclosure_levels(self):
        """Test that @op supports progressive disclosure."""

        # Level 1: Simple function
        @op
        def simple(x):
            return x * 2

        # Can be used immediately
        assert simple(5) == 10

        # But is actually an Operator
        assert isinstance(simple, Operator)
        assert hasattr(simple, "forward")

        # Can be composed
        pipeline = chain(simple, simple, simple)  # 2^3 = 8x
        assert pipeline(5) == 40

        # Can be used with JAX (if it had arrays)
        # Can be optimized with XCS
        # All without changing the original simple function!


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
