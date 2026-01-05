"""Test that PRNG key batching works correctly with operators."""

from typing import List

import jax
import jax.numpy as jnp
import pytest

from ember._internal.module import Module
from ember.xcs import vmap


class RandomOperator(Module):
    """Operator that uses PRNG keys internally."""

    dim: int
    dropout_rate: float

    def __init__(self, dim: int, dropout_rate: float = 0.1):
        self.dim = dim
        self.dropout_rate = dropout_rate

    def forward(self, x: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        # Split key for multiple random operations
        k1, k2 = jax.random.split(key)

        # Random dropout mask
        mask = jax.random.bernoulli(k1, 1 - self.dropout_rate, x.shape)
        x = x * mask / (1 - self.dropout_rate)

        # Random noise
        noise = jax.random.normal(k2, x.shape) * 0.01
        return x + noise


class NestedRandomOperator(Module):
    """Operator that contains other operators using randomness."""

    sub_ops: List[RandomOperator]

    def __init__(self, num_ops: int, dim: int):
        self.sub_ops = [RandomOperator(dim) for _ in range(num_ops)]

    def forward(self, x: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        # Split key for each sub-operator
        keys = jax.random.split(key, len(self.sub_ops))

        result = x
        for op, k in zip(self.sub_ops, keys, strict=False):
            result = op.forward(result, k)
        return result


def test_single_key_batching():
    """Test that single PRNG keys can be properly batched."""
    batch_size = 4
    dim = 3

    op = RandomOperator(dim)

    # Create batched inputs
    key = jax.random.PRNGKey(42)
    x_batch = jnp.ones((batch_size, dim))
    keys = jax.random.split(key, batch_size)

    # vmap the forward function
    vmapped_forward = vmap(op.forward)

    # This should work without "key array of shape (4, 2)" errors
    results = vmapped_forward(x_batch, keys)

    assert results.shape == (batch_size, dim)
    # Results should be different due to different random keys
    assert not jnp.allclose(results[0], results[1])


def test_nested_key_batching():
    """Test key batching with nested operators."""
    batch_size = 3
    dim = 4
    num_sub_ops = 2

    op = NestedRandomOperator(num_sub_ops, dim)

    key = jax.random.PRNGKey(42)
    x_batch = jnp.ones((batch_size, dim))
    keys = jax.random.split(key, batch_size)

    vmapped_forward = vmap(op.forward)
    results = vmapped_forward(x_batch, keys)

    assert results.shape == (batch_size, dim)
    # Each batch element should have different results
    for i in range(1, batch_size):
        assert not jnp.allclose(results[0], results[i])


def test_key_splitting_in_operator_creation():
    """Test that keys work correctly when creating operators in vmap."""
    batch_size = 4
    dim = 3

    def create_and_init(key: jax.random.PRNGKey) -> jnp.ndarray:
        # This pattern is common - using key to initialize operator
        op_key, forward_key = jax.random.split(key)

        # Create operator with random initialization
        weight = jax.random.normal(op_key, (dim, dim))

        class TempOp(Module):
            w: jnp.ndarray

            def __init__(self, weight):
                self.w = weight

            def forward(self, x, key):
                noise = jax.random.normal(key, x.shape) * 0.01
                return x @ self.w + noise

        op = TempOp(weight)
        x = jnp.ones(dim)
        return op.forward(x, forward_key)

    # Create batch of keys
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch_size)

    # vmap the entire operation
    vmapped_fn = vmap(create_and_init)

    # This should work - keys should be 2D going in, but each function gets 1D
    results = vmapped_fn(keys)

    assert results.shape == (batch_size, dim)
    # Different keys should give different results
    assert not jnp.allclose(results[0], results[1])


def test_mixed_key_and_array_batching():
    """Test batching with both arrays and keys."""

    class MixedOp(Module):
        scale: float

        def __init__(self, scale: float = 1.0):
            self.scale = scale

        def forward(self, x: jnp.ndarray, key: jax.random.PRNGKey, threshold: float) -> jnp.ndarray:
            # Use key for random mask
            mask = jax.random.bernoulli(key, threshold, x.shape)
            return x * mask * self.scale

    batch_size = 3
    dim = 5

    op = MixedOp(scale=2.0)

    key = jax.random.PRNGKey(42)
    x_batch = jnp.ones((batch_size, dim))
    keys = jax.random.split(key, batch_size)
    thresholds = jnp.array([0.3, 0.5, 0.7])

    # vmap over all inputs
    vmapped_forward = vmap(op.forward)
    results = vmapped_forward(x_batch, keys, thresholds)

    assert results.shape == (batch_size, dim)
    # Different thresholds should give different sparsity
    sparsity = [jnp.mean(r == 0.0) for r in results]
    assert sparsity[0] > sparsity[2]  # Higher threshold = fewer zeros (more ones)


def test_key_error_message():
    """Test that we get clear error messages for key shape issues."""
    dim = 3
    op = RandomOperator(dim)

    # Try to pass a 2D key array directly (not split)
    key = jax.random.PRNGKey(42)
    keys_2d = jax.random.split(key, 4)  # Shape (4, 2)
    x = jnp.ones(dim)

    # This should give a clear error about key shape
    with pytest.raises(ValueError) as exc_info:
        # Try to use 2D key array directly
        op.forward(x, keys_2d)

    assert "key array of shape" in str(exc_info.value)
    assert "Use jax.vmap for batching" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
