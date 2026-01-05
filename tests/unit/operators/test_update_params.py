"""Test the update_params method on Operator base class."""

from typing import Any, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ember.api import operators


class SimpleOperator(operators.Operator):
    """Test operator with learnable parameters."""

    # Dynamic fields (JAX arrays)
    weights: jax.Array
    bias: jax.Array

    # Static fields - containers must be explicitly marked
    name: str
    config: Dict[str, Any] = eqx.field(static=True)

    def __init__(self, dim: int):
        self.weights = jnp.ones(dim, dtype=jnp.float32)
        self.bias = jnp.zeros((), dtype=jnp.float32)
        self.name = "test_op"
        self.config = {"dim": dim}

    def forward(self, x):
        return jnp.dot(x, self.weights) + self.bias


def test_update_single_param():
    """Test updating a single parameter."""
    op = SimpleOperator(3)

    # Update weights
    new_weights = jnp.array([2.0, 3.0, 4.0])
    new_op = op.update_params(weights=new_weights)

    # Check weights updated
    assert jnp.allclose(new_op.weights, new_weights)

    # Check other params unchanged
    assert jnp.allclose(new_op.bias, op.bias)
    assert new_op.name == op.name
    assert new_op.config == op.config

    # Check original unchanged (immutability)
    assert jnp.allclose(op.weights, jnp.ones(3))


def test_update_multiple_params():
    """Test updating multiple parameters at once."""
    op = SimpleOperator(3)

    # Update weights and bias
    new_weights = jnp.array([2.0, 3.0, 4.0])
    new_bias = jnp.array(1.5)
    new_op = op.update_params(weights=new_weights, bias=new_bias)

    # Check both updated
    assert jnp.allclose(new_op.weights, new_weights)
    assert jnp.allclose(new_op.bias, new_bias)

    # Check others unchanged
    assert new_op.name == op.name
    assert new_op.config == op.config


def test_gradient_update_pattern():
    """Test the gradient descent pattern with update_params."""
    op = SimpleOperator(3)

    # Define a simple loss function
    def loss_fn(operator, x):
        return jnp.sum(operator(x) ** 2)

    # Compute gradients
    x = jnp.array([1.0, 2.0, 3.0])
    grads = jax.grad(loss_fn)(op, x)

    # Update with gradients
    learning_rate = 0.01
    new_op = op.update_params(
        weights=op.weights - learning_rate * grads.weights,
        bias=op.bias - learning_rate * grads.bias,
    )

    # Check loss decreased
    old_loss = loss_fn(op, x)
    new_loss = loss_fn(new_op, x)
    assert new_loss < old_loss


def test_optax_integration():
    """Test integration with optax optimizers."""
    try:
        import optax
    except ImportError:
        pytest.skip("optax not installed")

    op = SimpleOperator(3)

    # Define loss
    def loss_fn(operator, x):
        return jnp.sum(operator(x) ** 2)

    # Initialize optimizer
    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(op)

    # Training step
    x = jnp.array([1.0, 2.0, 3.0])
    grads = jax.grad(loss_fn)(op, x)

    # Get updates from optimizer
    updates, opt_state = optimizer.update(grads, opt_state)

    # Apply updates using update_params
    new_op = op.update_params(weights=op.weights + updates.weights, bias=op.bias + updates.bias)

    # Check parameters changed
    assert not jnp.allclose(new_op.weights, op.weights)
    assert not jnp.allclose(new_op.bias, op.bias)


def test_vmap_with_update_params():
    """Test that update_params works with vmap."""
    op = SimpleOperator(3)

    # Create batch of new weights
    batch_weights = jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])

    # vmap the update operation
    def update_op_weights(weights):
        return op.update_params(weights=weights)

    # This should work without errors
    vmapped_update = jax.vmap(update_op_weights)
    new_ops = vmapped_update(batch_weights)

    # Check we got a batch of operators
    assert new_ops.weights.shape == (3, 3)


def test_jit_with_update_params():
    """Test that update_params works with JIT compilation."""
    op = SimpleOperator(3)

    @jax.jit
    def train_step(operator, x, learning_rate):
        # Compute loss and gradients
        def loss_fn(op):
            return jnp.sum(op(x) ** 2)

        grads = jax.grad(loss_fn)(operator)

        # Update parameters
        return operator.update_params(
            weights=operator.weights - learning_rate * grads.weights,
            bias=operator.bias - learning_rate * grads.bias,
        )

    # This should compile and run without errors
    x = jnp.array([1.0, 2.0, 3.0])
    new_op = train_step(op, x, 0.01)

    # Check parameters updated
    assert not jnp.allclose(new_op.weights, op.weights)
    assert not jnp.allclose(new_op.bias, op.bias)


def test_update_nonexistent_param():
    """Test that updating non-existent parameters raises error."""
    op = SimpleOperator(3)

    # This should raise an AttributeError
    with pytest.raises(AttributeError):
        op.update_params(nonexistent=jnp.array([1, 2, 3]))


def test_update_preserves_operator_type():
    """Test that update_params preserves the operator type."""
    op = SimpleOperator(3)

    # Update weights
    new_weights = jnp.array([2.0, 3.0, 4.0])
    new_op = op.update_params(weights=new_weights)

    # Check type preserved
    assert isinstance(new_op, SimpleOperator)
    assert isinstance(new_op, type(op))


if __name__ == "__main__":
    # Run basic tests
    test_update_single_param()
    test_update_multiple_params()
    test_gradient_update_pattern()
    test_optax_integration()
    test_vmap_with_update_params()
    test_jit_with_update_params()
    test_update_preserves_operator_type()

    print("✅ All update_params tests passed!")
