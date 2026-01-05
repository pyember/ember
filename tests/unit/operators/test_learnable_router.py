"""Tests for the LearnableRouter operator."""

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from ember.operators import LearnableRouter, Operator

# Global call tracker for testing
_lr_call_tracker = {}


class MockOperator(Operator):
    """Mock operator for testing."""

    result: Any

    def __init__(self, result):
        self.result = result
        _lr_call_tracker[id(self)] = 0

    def forward(self, input):
        _lr_call_tracker[id(self)] += 1
        return self.result

    @property
    def call_count(self):
        return _lr_call_tracker.get(id(self), 0)


class TestLearnableRouter:
    """Test ML-based routing operator."""

    def test_learnable_router_creation(self):
        """Test creating a learnable router."""
        op1 = MockOperator("result1")
        op2 = MockOperator("result2")
        op3 = MockOperator("result3")

        # LearnableRouter requires dimensions and key
        key = jax.random.PRNGKey(42)
        router = LearnableRouter(
            routes={"route1": op1, "route2": op2, "route3": op3}, embed_dim=10, key=key
        )

        # Should have routing weights
        assert hasattr(router, "routing_weights")
        assert hasattr(router, "route_names")
        assert hasattr(router, "temperature")

        # Routing weights should have correct shape
        assert router.routing_weights.shape == (10, 3)

    def test_learnable_router_forward(self):
        """Test routing with learned weights."""
        op1 = MockOperator("A")
        op2 = MockOperator("B")
        op3 = MockOperator("C")

        key = jax.random.PRNGKey(42)

        # Define embedding function
        def embed_fn(x):
            # Simple embedding based on input
            return jnp.array([float(ord(x[0])), 0.0, 0.0, 0.0])

        router = LearnableRouter(
            routes={"r1": op1, "r2": op2, "r3": op3},
            embed_dim=4,
            key=key,
            embedding_fn=embed_fn,
        )

        # Route based on input
        result = router("test")

        # Should route to one of the operators
        assert result in ["A", "B", "C"]

        # Exactly one operator should be called
        total_calls = op1.call_count + op2.call_count + op3.call_count
        assert total_calls == 1

    def test_learnable_router_different_embeddings(self):
        """Test that different embeddings route to operators."""
        op1 = MockOperator("first")
        op2 = MockOperator("second")

        key = jax.random.PRNGKey(42)

        # Without embedding function, expects structured input
        router = LearnableRouter(
            routes={"r1": op1, "r2": op2},
            embed_dim=2,
            key=key,
            embedding_fn=None,  # Expects input with .embedding
        )

        # Create input with embedding
        from dataclasses import dataclass

        @dataclass
        class RoutingInput:
            data: str
            embedding: jax.Array

        # Different embeddings
        input1 = RoutingInput("data1", jnp.array([1.0, 0.0]))
        input2 = RoutingInput("data2", jnp.array([0.0, 1.0]))

        # Route based on embeddings
        result1 = router(input1)
        result2 = router(input2)

        # Should route to one of the operators
        assert result1 in ["first", "second"]
        assert result2 in ["first", "second"]

    def test_learnable_router_jax_compatibility(self):
        """Test that router weights are JAX arrays."""
        key = jax.random.PRNGKey(42)
        router = LearnableRouter(
            routes={"a": MockOperator(1), "b": MockOperator(2)}, embed_dim=3, key=key
        )

        # Weights should be JAX arrays
        assert isinstance(router.routing_weights, jnp.ndarray)
        assert isinstance(router.temperature, jnp.ndarray)

        # Can compute routing probabilities
        embedding = jnp.ones(3)
        probs = router.compute_route_probabilities(embedding)

        # Should be valid probabilities
        assert probs.shape == (2,)
        assert jnp.allclose(probs.sum(), 1.0)

    def test_learnable_router_probabilities_jitted(self):
        """compute_route_probabilities should work under jax.jit."""
        key = jax.random.PRNGKey(0)
        router = LearnableRouter(
            routes={"x": MockOperator(0), "y": MockOperator(1)}, embed_dim=2, key=key
        )

        embedding = jnp.array([0.5, -0.5])

        @jax.jit
        def jitted_probs(e):
            return router.compute_route_probabilities(e)

        probs = jitted_probs(embedding)
        assert probs.shape == (2,)
        assert jnp.allclose(jnp.sum(probs), 1.0)

    def test_learnable_router_forward_index_conversion(self):
        """Forward path should safely convert JAX index to Python int for list lookup."""
        # Two routes whose outputs are distinct values; embed_fn deterministically routes
        op_left = MockOperator("L")
        op_right = MockOperator("R")

        key = jax.random.PRNGKey(123)

        def embed_fn(x):
            # Map first character ordinal to a 2D embedding; simple linear separation
            base = float(ord(str(x)[0]))
            return jnp.array([base, 1.0])

        router = LearnableRouter(
            routes={"left": op_left, "right": op_right},
            embed_dim=2,
            key=key,
            embedding_fn=embed_fn,
            temperature=1.0,
        )

        # Should not raise TypeError from indexing; should return one of the two values
        result = router("alpha")
        assert result in ("L", "R")

    def test_learnable_router_invalid_temperature_raises(self):
        """Construction should fail when temperature <= 0."""
        key = jax.random.PRNGKey(0)
        with pytest.raises(ValueError):
            LearnableRouter(
                routes={"a": MockOperator(1), "b": MockOperator(2)},
                embed_dim=2,
                key=key,
                temperature=0.0,
            )

    def test_learnable_router_weight_update(self):
        """Test that router weights are differentiable."""
        key = jax.random.PRNGKey(42)

        def embed_fn(x):
            return jnp.ones(2) * float(x)

        router = LearnableRouter(
            routes={"pos": MockOperator(1.0), "neg": MockOperator(-1.0)},
            embed_dim=2,
            key=key,
            embedding_fn=embed_fn,
        )

        # Define a simple loss using compute_route_probabilities
        def loss_fn(weights, embedding):
            # Create a new router with updated weights
            router_copy = LearnableRouter(
                routes=router.routes, embed_dim=2, key=key, embedding_fn=embed_fn
            )
            # Manually set weights (in practice use eqx.tree_at)
            object.__setattr__(router_copy, "routing_weights", weights)

            probs = router_copy.compute_route_probabilities(embedding)
            # Want to maximize probability of first route
            return -jnp.log(probs[0])

        # Compute gradient
        embedding = jnp.ones(2)
        grad = jax.grad(loss_fn)(router.routing_weights, embedding)

        # Gradient should have same shape as weights
        assert grad.shape == router.routing_weights.shape
