"""Test that nested operator structures don't produce JAX static warnings."""

import warnings
from typing import List

import jax
import jax.numpy as jnp
import pytest

from ember._internal.module import Module


class TensorOp(Module):
    """Operator with JAX arrays."""

    weight: jnp.ndarray

    def __init__(self, dim: int, key: jax.random.PRNGKey):
        self.weight = jax.random.normal(key, (dim, dim))


class NestedOp(Module):
    """Operator containing other operators."""

    name: str
    ops: List[TensorOp]

    def __init__(self, name: str, num_ops: int, dim: int, key: jax.random.PRNGKey):
        self.name = name
        keys = jax.random.split(key, num_ops)
        self.ops = [TensorOp(dim, keys[i]) for i in range(num_ops)]


class DeeplyNestedOp(Module):
    """Multiple levels of nesting."""

    config: dict
    branches: List[NestedOp]

    def __init__(self, num_branches: int, ops_per_branch: int, dim: int, key: jax.random.PRNGKey):
        self.config = {"branches": num_branches, "ops": ops_per_branch}
        keys = jax.random.split(key, num_branches)
        self.branches = [
            NestedOp(f"branch_{i}", ops_per_branch, dim, keys[i]) for i in range(num_branches)
        ]


def test_no_warnings_for_nested_operators():
    """Nested operators with JAX arrays should not produce static warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        key = jax.random.PRNGKey(42)
        op = DeeplyNestedOp(num_branches=3, ops_per_branch=4, dim=8, key=key)

        # Check that the operator was created successfully
        assert len(op.branches) == 3
        assert len(op.branches[0].ops) == 4
        assert op.branches[0].ops[0].weight.shape == (8, 8)

        # Filter to only JAX static warnings
        static_warnings = [
            warning for warning in w if "JAX array is being set as static" in str(warning.message)
        ]

        # This test should FAIL with current implementation
        assert len(static_warnings) == 0, f"Got {len(static_warnings)} static warnings"


def test_jit_with_nested_operators():
    """JIT compilation should work correctly with nested operators."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        key = jax.random.PRNGKey(42)
        op = NestedOp("test", num_ops=2, dim=4, key=key)

        @jax.jit
        def forward(op, x):
            # Use the first operator's weight
            return x @ op.ops[0].weight

        x = jnp.ones(4)
        result = forward(op, x)

        # Should work without warnings
        static_warnings = [
            warning for warning in w if "JAX array is being set as static" in str(warning.message)
        ]

        assert len(static_warnings) == 0
        assert result.shape == (4,)


def test_mixed_static_dynamic_fields():
    """Operators with both static config and dynamic arrays should work."""

    class MixedOp(Module):
        # Static fields
        name: str
        config: dict
        threshold: float

        # Dynamic fields
        weight: jnp.ndarray
        bias: jnp.ndarray

        # Container that might have dynamic content
        sub_ops: List[TensorOp]

        def __init__(self, name: str, dim: int, num_sub: int, key: jax.random.PRNGKey):
            # Static
            self.name = name
            self.config = {"dim": dim, "num_sub": num_sub}
            self.threshold = 0.5

            # Dynamic
            w_key, b_key, *sub_keys = jax.random.split(key, 2 + num_sub)
            self.weight = jax.random.normal(w_key, (dim, dim))
            self.bias = jax.random.normal(b_key, (dim,))

            # Mixed container
            self.sub_ops = [TensorOp(dim, sub_keys[i]) for i in range(num_sub)]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        key = jax.random.PRNGKey(42)
        op = MixedOp("mixed", dim=4, num_sub=2, key=key)

        # Verify structure
        assert op.name == "mixed"  # Should be static
        assert op.weight.shape == (4, 4)  # Should be dynamic
        assert len(op.sub_ops) == 2  # Container with dynamic content

        # No warnings
        static_warnings = [
            warning for warning in w if "JAX array is being set as static" in str(warning.message)
        ]
        assert len(static_warnings) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
