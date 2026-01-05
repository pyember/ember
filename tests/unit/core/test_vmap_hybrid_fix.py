"""Test that vmap works correctly with hybrid tensor/orchestration operations."""

from typing import List, Tuple

import jax
import jax.numpy as jnp
import pytest

from ember._internal.module import Module
from ember.xcs import vmap


class HybridOperator(Module):
    """Operator that mixes tensor and orchestration operations."""

    operation_name: str
    weight: jnp.ndarray

    def __init__(self, dim: int, key: jax.random.PRNGKey):
        self.operation_name = "hybrid_op"
        self.weight = jax.random.normal(key, (dim, dim))

    def forward(self, tensor_input: jnp.ndarray, text_input: str) -> Tuple[jnp.ndarray, str]:
        # Tensor operation
        tensor_output = jnp.tanh(tensor_input @ self.weight)

        # Orchestration operation (simulated)
        text_output = f"processed_{text_input}"

        return tensor_output, text_output


def test_vmap_with_hybrid_operator():
    """vmap should correctly handle operators with both tensor and orchestration ops."""
    batch_size = 4
    dim = 3

    # Create batched inputs
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch_size)
    tensor_batch = jax.random.normal(key, (batch_size, dim))
    text_batch = [f"item_{i}" for i in range(batch_size)]

    def create_and_run(tensor_x: jnp.ndarray, text_x: str, key: jax.random.PRNGKey):
        op = HybridOperator(dim, key)
        return op.forward(tensor_x, text_x)

    # Apply vmap
    vmapped_fn = vmap(create_and_run)

    # This should work without errors
    tensor_results, text_results = vmapped_fn(tensor_batch, text_batch, keys)

    # Verify results
    assert tensor_results.shape == (batch_size, dim)
    assert len(text_results) == batch_size
    assert all(isinstance(r, str) for r in text_results)
    assert all(r.startswith("processed_item_") for r in text_results)


def test_vmap_key_splitting():
    """vmap should correctly split PRNG keys for batched operations."""
    batch_size = 3
    dim = 2

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch_size)

    def create_and_get_weight(key: jax.random.PRNGKey) -> jnp.ndarray:
        """Create operator and return its weight."""
        op = HybridOperator(dim, key)
        return op.weight

    # vmap over weight extraction
    vmapped_fn = vmap(create_and_get_weight)

    # This should work - keys should be properly split
    weights = vmapped_fn(keys)

    # Check that we got different weights for each key
    assert weights.shape == (batch_size, dim, dim)

    # Weights should be different (very unlikely to be equal with random init)
    assert not jnp.allclose(weights[0], weights[1])
    assert not jnp.allclose(weights[1], weights[2])


def test_vmap_nested_operators():
    """vmap should work with nested operator structures."""

    class OuterOp(Module):
        name: str
        inner_ops: List[HybridOperator]  # Properly typed

        def __init__(self, name: str, num_inner: int, dim: int, key: jax.random.PRNGKey):
            self.name = name
            keys = jax.random.split(key, num_inner)
            self.inner_ops = [HybridOperator(dim, keys[i]) for i in range(num_inner)]

        def forward(self, x: jnp.ndarray) -> jnp.ndarray:
            # Use first inner op
            result, _ = self.inner_ops[0].forward(x, "test")
            return result

    batch_size = 2
    dim = 3

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch_size)
    x_batch = jax.random.normal(key, (batch_size, dim))

    def create_and_run(x: jnp.ndarray, key: jax.random.PRNGKey):
        op = OuterOp("outer", num_inner=2, dim=dim, key=key)
        return op.forward(x)

    # This should work without issues
    vmapped_fn = vmap(create_and_run)
    results = vmapped_fn(x_batch, keys)

    assert results.shape == (batch_size, dim)


def test_vmap_in_axes_specification():
    """vmap should respect in_axes specifications for hybrid operations."""
    dim = 4
    key = jax.random.PRNGKey(42)
    op = HybridOperator(dim, key)

    # Batch over first axis of tensor, but not text
    tensor_batch = jax.random.normal(key, (3, dim))
    single_text = "shared_text"

    # vmap only over tensor input
    vmapped_forward = vmap(op.forward, in_axes=(0, None))

    tensor_results, text_results = vmapped_forward(tensor_batch, single_text)

    assert tensor_results.shape == (3, dim)
    # All text results should be the same
    assert len(set(text_results)) == 1
    assert text_results[0] == "processed_shared_text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
