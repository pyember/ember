"""Test that model bindings and orchestration work correctly with static fields."""

import warnings
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import pytest

from ember._internal.module import Module


class OrchestrationModule(Module):
    """Pure orchestration module with model/tool references."""

    name: str
    model: str  # Model identifier
    tools: List[str]  # Tool names
    config: Dict[str, Any]

    def __init__(self, name: str):
        self.name = name
        self.model = "gpt-4"
        self.tools = ["web_search", "calculator", "code_interpreter"]
        self.config = {"temperature": 0.7, "max_retries": 3, "timeout": 30}

    def call_model(self, prompt: str) -> str:
        """Simulate model API call."""
        return f"{self.model}({prompt})"

    def call_tool(self, tool_idx: int, args: str) -> str:
        """Simulate tool call."""
        return f"{self.tools[tool_idx]}({args})"


class HybridModule(Module):
    """Module with both orchestration config and learnable parameters."""

    # Orchestration configuration (should be static)
    model_primary: str
    model_fallback: str
    routing_strategy: str

    # Learnable parameters (should be dynamic)
    routing_weights: jnp.ndarray
    embedding: jnp.ndarray

    def __init__(self, dim: int, key: jax.random.PRNGKey):
        # Static orchestration config
        self.model_primary = "claude-3"
        self.model_fallback = "gpt-4"
        self.routing_strategy = "confidence_based"

        # Dynamic learnable parameters
        k1, k2 = jax.random.split(key)
        self.routing_weights = jax.nn.softmax(jax.random.normal(k1, (2,)))
        self.embedding = jax.random.normal(k2, (dim,))

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        # Use embedding for computation
        return x @ self.embedding


def test_orchestration_module_creation():
    """Test that orchestration modules are created without warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        module = OrchestrationModule("test")

        # No static array warnings
        static_warnings = [
            warning for warning in w if "JAX array is being set as static" in str(warning.message)
        ]
        assert len(static_warnings) == 0

        # Fields are accessible
        assert module.model == "gpt-4"
        assert len(module.tools) == 3
        assert module.config["temperature"] == 0.7


def test_hybrid_module_partitioning():
    """Test that hybrid modules correctly partition static/dynamic."""
    key = jax.random.PRNGKey(42)
    module = HybridModule(dim=5, key=key)

    # Get pytree structure
    leaves, treedef = jax.tree_util.tree_flatten(module)

    # Count JAX arrays (dynamic)
    jax_arrays = [leaf for leaf in leaves if isinstance(leaf, jnp.ndarray)]

    # Should have exactly 2 arrays: routing_weights and embedding
    assert len(jax_arrays) == 2

    # Check shapes
    shapes = [arr.shape for arr in jax_arrays]
    assert (2,) in shapes  # routing_weights
    assert (5,) in shapes  # embedding


def test_jit_compilation_with_orchestration():
    """Test JIT compilation with orchestration modules."""
    module = OrchestrationModule("test")

    @jax.jit
    def simulate_orchestration(module):
        # Simulate accessing config during execution
        if hasattr(module, "config"):
            temp = module.config.get("temperature", 0.5)
        else:
            temp = 0.5

        # Return a JAX array based on config
        return jnp.array([temp])

    # Should compile without issues
    result = simulate_orchestration(module)
    assert result[0] == 0.7


def test_vmap_with_orchestration():
    """Test that vmap works with orchestration modules."""
    from ember.xcs import vmap

    batch_size = 3
    key = jax.random.PRNGKey(42)

    # Create a single module
    module = HybridModule(dim=4, key=key)

    # Create batched input
    x_batch = jnp.ones((batch_size, 4))

    # vmap the forward function
    vmapped_forward = vmap(module.forward)

    # Apply to batch
    results = vmapped_forward(x_batch)

    # The forward method returns x @ embedding which is (4,) @ (4,) = scalar
    # So vmapped should return (batch_size,)
    assert results.shape == (batch_size,)

    # Also test that the module's static fields remain accessible
    assert module.model_primary == "claude-3"
    assert module.model_fallback == "gpt-4"


def test_no_recompilation_static_changes():
    """Test that changing static orchestration config triggers recompilation."""

    class SimpleModule(Module):
        model: str
        value: jnp.ndarray

        def __init__(self, model: str, value: float):
            self.model = model
            self.value = jnp.array(value)

        def forward(self, x):
            return x * self.value

    # Track compilations using side effects
    compilation_count = 0

    def traced_fn(m, x):
        nonlocal compilation_count
        compilation_count += 1
        return m.forward(x)

    # Create a custom jit that tracks compilations
    from functools import wraps

    @wraps(traced_fn)
    def jitted_fn(m, x):
        # Use hash of static fields to track recompilations
        static_key = (m.model,)  # Only model is static
        if not hasattr(jitted_fn, "_cache"):
            jitted_fn._cache = {}

        if static_key not in jitted_fn._cache:
            jitted_fn._cache[static_key] = jax.jit(lambda m, x: traced_fn(m, x))

        return jitted_fn._cache[static_key](m, x)

    # First module
    m1 = SimpleModule("gpt-4", 1.0)
    x = jnp.array(2.0)

    compilation_count = 0
    jitted_fn(m1, x)
    first_count = compilation_count
    assert first_count > 0  # Should compile

    # Same static config, different dynamic value
    m2 = SimpleModule("gpt-4", 2.0)
    jitted_fn(m2, x)
    assert compilation_count == first_count  # No recompilation

    # Different static config
    m3 = SimpleModule("claude-3", 1.0)
    jitted_fn(m3, x)
    assert compilation_count > first_count  # Should recompile


def test_complex_orchestration_graph():
    """Test a realistic orchestration graph structure."""

    class OrchestrationGraph(Module):
        """Represents a complex orchestration workflow."""

        nodes: Dict[str, str]  # node_id -> model/tool
        edges: Dict[str, List[str]]  # adjacency list
        node_configs: Dict[str, dict]  # per-node config

        def __init__(self):
            # Define the orchestration graph
            self.nodes = {
                "start": "input_processor",
                "analyze": "gpt-4",
                "search": "web_search",
                "synthesize": "claude-3",
                "verify": "fact_checker",
                "output": "formatter",
            }

            self.edges = {
                "start": ["analyze"],
                "analyze": ["search", "synthesize"],
                "search": ["synthesize"],
                "synthesize": ["verify"],
                "verify": ["output"],
                "output": [],
            }

            self.node_configs = {
                "analyze": {"temperature": 0.3, "max_tokens": 500},
                "synthesize": {"temperature": 0.7, "max_tokens": 1000},
                "verify": {"threshold": 0.8},
            }

        def get_execution_order(self) -> List[str]:
            """Topological sort for execution order."""
            # Simplified - just return a valid order
            return ["start", "analyze", "search", "synthesize", "verify", "output"]

    # Create without warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        graph = OrchestrationGraph()
        order = graph.get_execution_order()

        # No warnings
        static_warnings = [
            warning for warning in w if "JAX array is being set as static" in str(warning.message)
        ]
        assert len(static_warnings) == 0

        # Correct structure
        assert len(order) == 6
        assert order[0] == "start"
        assert order[-1] == "output"


def test_model_switching_recompilation():
    """Test that model switching behavior works correctly."""

    class AdaptiveOrchestrator(Module):
        """Orchestrator that can switch models."""

        primary_model: str
        fallback_model: str
        confidence_threshold: float

        # Dynamic confidence tracking
        confidence_history: jnp.ndarray

        def __init__(self, key: jax.random.PRNGKey):
            self.primary_model = "claude-3-opus"
            self.fallback_model = "gpt-4-turbo"
            self.confidence_threshold = 0.8

            # Track last 10 confidence scores
            self.confidence_history = jnp.zeros(10)

        def should_use_fallback(self) -> jnp.ndarray:
            """Check if we should switch to fallback."""
            avg_confidence = jnp.mean(self.confidence_history)
            # Return as array for JAX compatibility
            return jnp.array(avg_confidence < self.confidence_threshold)

    key = jax.random.PRNGKey(42)
    orchestrator = AdaptiveOrchestrator(key)

    @jax.jit
    def check_fallback(orch):
        return orch.should_use_fallback()

    # Initial check
    needs_fallback = check_fallback(orchestrator)
    assert bool(needs_fallback)  # Low confidence initially

    # Create new orchestrator with high confidence
    import equinox as eqx

    new_confidence = jnp.ones(10) * 0.9
    orchestrator_high = eqx.tree_at(lambda x: x.confidence_history, orchestrator, new_confidence)

    needs_fallback = check_fallback(orchestrator_high)
    assert not bool(needs_fallback)  # High confidence now


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
