"""Integration tests for JAX transformations with Ember operators.

Tests the integration between Ember operators and JAX transformations,
ensuring they work together correctly in realistic scenarios.
"""

from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ember.api.operators import Operator


class RealisticOperator(Operator):
    """A realistic operator mixing various field types."""

    # Dynamic fields (participate in JAX transformations)
    embedding_matrix: jax.Array
    projection_weights: jax.Array
    temperature: jax.Array

    # Static fields (configuration)
    vocab_size: int
    hidden_dim: int
    config: Dict[str, any]

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 16):
        # Initialize dynamic parameters
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        self.embedding_matrix = jax.random.normal(k1, (vocab_size, hidden_dim)) * 0.1
        self.projection_weights = jax.random.normal(k2, (hidden_dim, hidden_dim)) * 0.1
        self.temperature = jnp.array(1.0)

        # Initialize static configuration
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.config = {"activation": "relu", "dropout_rate": 0.1, "normalize": True}

    def forward(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        """Process token IDs through embedding and projection."""
        # Embedding lookup
        embeddings = self.embedding_matrix[token_ids]

        # Optional normalization based on config
        if self.config["normalize"]:
            embeddings = embeddings / jnp.linalg.norm(embeddings, axis=-1, keepdims=True)

        # Project through weights
        projected = jnp.dot(embeddings, self.projection_weights)

        # Apply activation
        if self.config["activation"] == "relu":
            projected = jax.nn.relu(projected)

        # Temperature scaling
        return projected / self.temperature


class TestGradientIntegration:
    """Test gradient computation through operators."""

    def test_gradients_flow_through_operator_fields(self):
        """Gradients should flow only through dynamic fields."""
        # Arrange
        op = RealisticOperator(vocab_size=50, hidden_dim=8)
        tokens = jnp.array([1, 2, 3, 4])

        def loss_fn(operator):
            output = operator(tokens)
            return jnp.mean(output**2)

        # Act
        loss_value = loss_fn(op)
        assert loss_value >= 0
        grads = jax.grad(loss_fn)(op)

        # Assert - gradients exist for dynamic fields
        assert grads.embedding_matrix is not None
        assert grads.projection_weights is not None
        assert grads.temperature is not None

        # Assert - gradients have correct shapes
        assert grads.embedding_matrix.shape == op.embedding_matrix.shape
        assert grads.projection_weights.shape == op.projection_weights.shape
        assert grads.temperature.shape == op.temperature.shape

        # Assert - static fields preserved
        assert grads.vocab_size == op.vocab_size
        assert grads.config == op.config

    def test_gradient_based_optimization_works(self):
        """Should be able to optimize operator parameters."""
        # Arrange
        op = RealisticOperator(vocab_size=10, hidden_dim=4)
        tokens = jnp.array([1, 2, 3])
        target = jnp.ones((3, 4)) * 0.5  # Target output

        def loss_fn(operator):
            output = operator(tokens)
            return jnp.mean((output - target) ** 2)

        # Act - gradient descent step
        initial_loss = loss_fn(op)
        grads = jax.grad(loss_fn)(op)

        # Update parameters
        learning_rate = 0.1
        updated_op = op.update_params(
            embedding_matrix=op.embedding_matrix - learning_rate * grads.embedding_matrix,
            projection_weights=op.projection_weights - learning_rate * grads.projection_weights,
            temperature=op.temperature - learning_rate * grads.temperature,
        )

        final_loss = loss_fn(updated_op)

        # Assert
        assert final_loss < initial_loss, "Loss should decrease after gradient step"
        assert updated_op.config == op.config, "Static config should be unchanged"


class TestVmapIntegration:
    """Test vectorization over operators and their inputs."""

    def test_vmap_over_operator_inputs(self):
        """vmap should correctly batch operator forward passes."""
        # Arrange
        op = RealisticOperator(vocab_size=20, hidden_dim=8)
        batch_tokens = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])  # Shape: (4, 3)

        # Act - vmap over batch dimension
        vmapped_forward = jax.vmap(op.forward)
        batch_output = vmapped_forward(batch_tokens)

        # Assert
        assert batch_output.shape == (4, 3, 8)  # (batch, seq, hidden)

        # Verify against manual loop
        manual_output = jnp.stack([op(tokens) for tokens in batch_tokens])
        np.testing.assert_allclose(batch_output, manual_output, rtol=1e-5)

    def test_vmap_over_operators_with_different_parameters(self):
        """vmap should work when mapping over operator parameters."""
        # Arrange
        base_op = RealisticOperator(vocab_size=10, hidden_dim=4)
        tokens = jnp.array([1, 2, 3])

        # Create batch of different temperatures
        temperatures = jnp.array([0.5, 1.0, 2.0, 4.0])

        # Act - vmap over temperature parameter
        def forward_with_temp(temp):
            return base_op.update_params(temperature=temp)(tokens)

        vmapped_forward = jax.vmap(forward_with_temp)
        outputs = vmapped_forward(temperatures)

        # Assert
        assert outputs.shape == (4, 3, 4)  # (num_temps, seq_len, hidden_dim)

        # Verify temperature scaling works
        # Higher temperature should produce smaller magnitude outputs
        output_norms = jnp.linalg.norm(outputs, axis=(1, 2))
        assert output_norms[0] > output_norms[-1]  # temp=0.5 > temp=4.0


class TestJitIntegration:
    """Test JIT compilation of operators."""

    def test_jit_compilation_preserves_behavior(self):
        """JIT compilation should not change operator behavior."""
        # Arrange
        op = RealisticOperator(vocab_size=30, hidden_dim=16)
        tokens = jnp.array([5, 10, 15, 20])

        # Act
        normal_output = op(tokens)

        # JIT compile the operator's forward method
        @jax.jit
        def jitted_forward(tokens):
            return op(tokens)

        jitted_output = jitted_forward(tokens)

        # Assert
        np.testing.assert_allclose(normal_output, jitted_output, rtol=1e-5)

    def test_jit_with_static_arguments(self):
        """JIT should handle operators as static arguments correctly."""
        # Arrange
        op = RealisticOperator(vocab_size=15, hidden_dim=8)

        # Define function with operator as static argument
        @jax.jit
        def process_tokens(tokens, static_scale=1.0):
            output = op(tokens)
            return output * static_scale

        # Act
        tokens1 = jnp.array([1, 2, 3])
        tokens2 = jnp.array([4, 5, 6])

        output1 = process_tokens(tokens1, static_scale=2.0)
        output2 = process_tokens(tokens2, static_scale=2.0)

        # Assert
        assert output1.shape == (3, 8)
        assert output2.shape == (3, 8)
        assert not jnp.array_equal(output1, output2)  # Different inputs


class TestComposedTransformations:
    """Test combinations of JAX transformations."""

    def test_grad_of_vmapped_operator(self):
        """Gradient of a vmapped operator execution."""
        # Arrange
        op = RealisticOperator(vocab_size=10, hidden_dim=4)
        batch_tokens = jnp.array([[1, 2], [3, 4], [5, 6]])

        def batch_loss(operator):
            # vmap the forward pass
            outputs = jax.vmap(operator.forward)(batch_tokens)
            return jnp.mean(outputs**2)

        # Act
        loss = batch_loss(op)
        grads = jax.grad(batch_loss)(op)

        # Assert
        assert isinstance(loss, jax.Array)
        assert grads.embedding_matrix is not None
        assert grads.embedding_matrix.shape == op.embedding_matrix.shape

    def test_jit_of_grad_computation(self):
        """JIT compilation of gradient computation."""
        # Arrange
        op = RealisticOperator(vocab_size=8, hidden_dim=4)
        tokens = jnp.array([1, 2, 3, 4])

        def loss_fn(operator):
            return jnp.sum(operator(tokens) ** 2)

        # Act
        # JIT the gradient computation
        jitted_grad = jax.jit(jax.grad(loss_fn))

        # Compute gradients
        grads = jitted_grad(op)

        # Assert
        assert grads.embedding_matrix is not None
        assert grads.projection_weights is not None

        # Compare with non-jitted version
        normal_grads = jax.grad(loss_fn)(op)
        np.testing.assert_allclose(grads.embedding_matrix, normal_grads.embedding_matrix, rtol=1e-5)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_batch_handling(self):
        """Operators should handle empty batches gracefully."""
        # Arrange
        op = RealisticOperator(vocab_size=10, hidden_dim=4)
        empty_tokens = jnp.array([], dtype=jnp.int32)

        # Act
        output = op(empty_tokens)

        # Assert
        assert output.shape == (0, 4)  # Empty batch, hidden_dim

    def test_operator_with_no_array_fields(self):
        """Operators without array fields should still work with JAX."""

        class StaticOperator(Operator):
            multiplier: float

            def __init__(self, multiplier: float = 2.0):
                self.multiplier = multiplier

            def forward(self, x):
                return x * self.multiplier

        # Arrange
        op = StaticOperator(3.0)

        # Act - should work with jit
        jitted = jax.jit(op)
        result = jitted(jnp.array([1.0, 2.0, 3.0]))

        # Assert
        np.testing.assert_allclose(result, jnp.array([3.0, 6.0, 9.0]))

    def test_nested_operators(self):
        """Operators containing other operators should work."""

        class ComposedOperator(Operator):
            encoder: RealisticOperator
            scale: jax.Array

            def __init__(self):
                self.encoder = RealisticOperator(vocab_size=10, hidden_dim=8)
                self.scale = jnp.array(0.1)

            def forward(self, tokens):
                encoded = self.encoder(tokens)
                return encoded * self.scale

        # Arrange
        op = ComposedOperator()
        tokens = jnp.array([1, 2, 3])

        # Act - test with transformations
        output = op(tokens)
        grads = jax.grad(lambda o: jnp.sum(o(tokens) ** 2))(op)

        # Assert
        assert output.shape == (3, 8)
        assert grads.scale is not None
        assert grads.encoder.embedding_matrix is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
