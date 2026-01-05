"""Test that model bindings and orchestration config remain static."""

from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import pytest

from ember._internal.module import Module


# Simulate model/tool bindings
class ModelBinding:
    """Represents a model binding."""

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    def __repr__(self):
        return f"{self.provider}.{self.model}"


class ToolBinding:
    """Represents a tool binding."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

    def __repr__(self):
        return f"Tool({self.name})"


class OrchestrationOperator(Module):
    """Pure orchestration operator with model/tool bindings."""

    name: str
    model: str  # String model reference
    tool: str  # String tool reference
    config: dict

    def __init__(self, name: str):
        self.name = name
        self.model = "gpt-4"  # Static
        self.tool = "web_search"  # Static
        self.config = {"temperature": 0.7, "max_tokens": 100, "retry_count": 3}

    def forward(self, prompt: str) -> str:
        # Simulate API call
        return f"{self.model}({prompt})"


class RouterOperator(Module):
    """Router with both static config and dynamic weights."""

    model_bindings: List[str]  # Static model references
    weights: jnp.ndarray  # Dynamic routing weights
    config: dict

    def __init__(self, models: List[str], key: jax.random.PRNGKey):
        self.model_bindings = models  # Should be static
        self.weights = jax.nn.softmax(jax.random.normal(key, (len(models),)))
        self.config = {"strategy": "weighted_random"}

    def forward(self, x: Any) -> str:
        # Route to model based on weights
        idx = jnp.argmax(self.weights)
        return f"{self.model_bindings[idx]}({x})"


class AdvancedOrchestrator(Module):
    """Complex orchestration with nested bindings."""

    name: str
    primary_model: ModelBinding
    fallback_model: ModelBinding
    tools: List[ToolBinding]
    orchestration_graph: Dict[str, List[str]]

    # Optional dynamic components
    adaptation_weights: jnp.ndarray

    def __init__(self, name: str, key: jax.random.PRNGKey):
        self.name = name
        self.primary_model = ModelBinding("openai", "gpt-4")
        self.fallback_model = ModelBinding("anthropic", "claude-2")
        self.tools = [
            ToolBinding("search", {"api_key": "xxx"}),
            ToolBinding("calculator", {"precision": "high"}),
        ]
        self.orchestration_graph = {
            "start": ["search", "primary_model"],
            "search": ["primary_model"],
            "primary_model": ["end"],
        }

        # Dynamic component for online adaptation
        self.adaptation_weights = jax.random.normal(key, (4,))


def test_orchestration_fields_are_static():
    """Test that orchestration config remains static during JIT."""
    op = OrchestrationOperator("test_op")

    # Test that fields are properly marked as static in equinox
    import equinox as eqx

    # Use eqx.is_array to partition arrays vs non-arrays
    dynamic_fields, static_fields = eqx.partition(op, eqx.is_array)

    # These should be in static fields
    assert hasattr(static_fields, "name")
    assert hasattr(static_fields, "model")
    assert hasattr(static_fields, "config")

    # Verify the values are preserved
    assert static_fields.name == "test_op"
    assert static_fields.model == "gpt-4"
    assert static_fields.config["temperature"] == 0.7

    # Dynamic fields should have no arrays for this operator
    leaves = jax.tree_util.tree_leaves(dynamic_fields)
    array_leaves = [leaf for leaf in leaves if isinstance(leaf, jnp.ndarray)]
    assert len(array_leaves) == 0  # No arrays in this operator


def test_jit_recompilation_behavior():
    """Test that changing static fields requires recompilation."""

    # Test that different static values cause recompilation
    trace_count = 0

    def process_with_model(model_name: str, x: jnp.ndarray):
        nonlocal trace_count
        trace_count += 1
        # Return a JAX array that depends on the model choice
        if model_name == "gpt-4":
            return x * 2.0
        else:
            return x * 3.0

    # Make model_name static
    jitted = jax.jit(process_with_model, static_argnums=(0,))

    # Reset counter
    trace_count = 0

    # First call with gpt-4
    x = jnp.array(1.0)
    result1 = jitted("gpt-4", x)
    assert result1 == 2.0
    assert trace_count == 1

    # Second call with same model - no recompilation
    result2 = jitted("gpt-4", x * 2)
    assert result2 == 4.0
    assert trace_count == 1  # No new trace

    # Call with different model - triggers recompilation
    result3 = jitted("claude-2", x)
    assert result3 == 3.0
    assert trace_count == 2  # New trace for different static arg


def test_mixed_static_dynamic_router():
    """Test router with both static bindings and dynamic weights."""
    key = jax.random.PRNGKey(42)
    models = ["gpt-4", "claude-2", "llama-2"]
    router = RouterOperator(models, key)

    # Check partitioning
    import equinox as eqx

    dynamic_router, static_router = eqx.partition(router, eqx.is_array)

    # Model bindings should be static
    assert hasattr(static_router, "model_bindings")
    assert static_router.model_bindings == models

    # Weights should be dynamic
    leaves = jax.tree_util.tree_leaves(dynamic_router)
    array_leaves = [leaf for leaf in leaves if isinstance(leaf, jnp.ndarray)]
    assert len(array_leaves) == 1  # Should have exactly one array (weights)
    assert array_leaves[0].shape == (3,)

    # Test that JIT works correctly
    @jax.jit
    def get_weights(router):
        return router.weights

    weights = get_weights(router)
    assert weights.shape == (3,)
    assert jnp.allclose(weights.sum(), 1.0)  # Softmax


def test_complex_orchestration_bindings():
    """Test complex nested orchestration structures."""
    key = jax.random.PRNGKey(42)
    orchestrator = AdvancedOrchestrator("main", key)

    # The complex nested structures should work
    assert orchestrator.primary_model.provider == "openai"
    assert len(orchestrator.tools) == 2
    assert "start" in orchestrator.orchestration_graph

    # Check static/dynamic partitioning with equinox
    import equinox as eqx

    dynamic_orch, static_orch = eqx.partition(orchestrator, eqx.is_array)

    # These should be static
    assert hasattr(static_orch, "name")
    assert hasattr(static_orch, "primary_model")
    assert hasattr(static_orch, "tools")
    assert hasattr(static_orch, "orchestration_graph")

    # Only adaptation weights should be dynamic
    leaves = jax.tree_util.tree_leaves(dynamic_orch)
    array_leaves = [leaf for leaf in leaves if isinstance(leaf, jnp.ndarray)]
    assert len(array_leaves) == 1  # Should have exactly one array
    assert array_leaves[0].shape == (4,)  # adaptation_weights

    # Test that we can JIT functions using the module
    @jax.jit
    def get_weights_only(weights):
        return jax.nn.softmax(weights)

    # Extract just the dynamic weights
    adapted = get_weights_only(orchestrator.adaptation_weights)
    assert adapted.shape == (4,)


def test_no_recompilation_for_dynamic_only_changes():
    """Test that changing dynamic fields doesn't trigger recompilation."""
    trace_count = 0

    class CountingOp(Module):
        config: dict  # Static
        weights: jnp.ndarray  # Dynamic

        def __init__(self, key):
            self.config = {"version": 1}
            self.weights = jax.random.normal(key, (3,))

        def forward(self, x):
            nonlocal trace_count
            trace_count += 1
            return x @ self.weights

    key = jax.random.PRNGKey(42)
    op1 = CountingOp(key)
    op2 = CountingOp(jax.random.split(key)[0])

    @jax.jit
    def apply(op, x):
        return op.forward(x)

    x = jnp.ones(3)

    # First call - traces
    trace_count = 0
    result1 = apply(op1, x)
    assert trace_count == 1

    # Second call with different weights - no retrace
    result2 = apply(op2, x)
    assert trace_count == 1  # No retracing
    assert not jnp.allclose(result1, result2)  # Different results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
