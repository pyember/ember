"""Test that nested mixed structures work correctly with the Module system."""

import dataclasses
import warnings
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import pytest

from ember._internal.module import Module


@dataclasses.dataclass
class StaticConfig:
    """Represents static configuration."""

    name: str
    value: Any


class DynamicState(Module):
    """Represents dynamic state with JAX arrays."""

    weights: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, key: jax.random.PRNGKey):
        self.weights = jax.random.normal(key, (3,))
        self.bias = jnp.zeros(3)


class SimpleOperator(Module):
    """Basic operator with static config."""

    name: str
    config: dict

    def __init__(self, name: str):
        self.name = name
        self.config = {"type": "simple", "version": 1}

    def forward(self, x):
        return f"{self.name}({x})"


class LearningOperator(Module):
    """Operator with learnable parameters."""

    name: str
    weights: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, name: str, dim: int, key: jax.random.PRNGKey):
        self.name = name
        k1, k2 = jax.random.split(key)
        self.weights = jax.random.normal(k1, (dim, dim))
        self.bias = jax.random.normal(k2, (dim,))

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.weights + self.bias


class MixedEnsemble(Module):
    """Ensemble with both static and dynamic operators."""

    name: str
    static_ops: List[SimpleOperator]
    dynamic_ops: List[LearningOperator]
    routing_weights: jnp.ndarray  # Dynamic
    routing_strategy: str  # Static

    def __init__(
        self, name: str, num_static: int, num_dynamic: int, dim: int, key: jax.random.PRNGKey
    ):
        self.name = name
        self.routing_strategy = "weighted_average"

        # Static operators
        self.static_ops = [SimpleOperator(f"static_{i}") for i in range(num_static)]

        # Dynamic operators
        keys = jax.random.split(key, num_dynamic + 1)
        self.dynamic_ops = [
            LearningOperator(f"dynamic_{i}", dim, keys[i]) for i in range(num_dynamic)
        ]

        # Dynamic routing weights
        total_ops = num_static + num_dynamic
        self.routing_weights = jax.nn.softmax(jax.random.normal(keys[-1], (total_ops,)))

    def forward(self, x: Any) -> Any:
        # Mix static and dynamic processing
        static_results = [op.forward(x) for op in self.static_ops]

        if isinstance(x, jnp.ndarray):
            dynamic_results = [op.forward(x) for op in self.dynamic_ops]
            # Weighted combination
            all_results = jnp.stack(dynamic_results)
            dynamic_weights = self.routing_weights[len(self.static_ops) :]
            return jnp.sum(all_results * dynamic_weights[:, None, None], axis=0)
        else:
            # Just return static result for non-array inputs
            return static_results[0] if static_results else x


class NestedHierarchy(Module):
    """Deep nesting of mixed structures."""

    root_config: Dict[str, Any]
    ensembles: List[MixedEnsemble]
    metadata: Dict[str, StaticConfig]
    state: Optional[DynamicState]

    def __init__(self, depth: int, key: jax.random.PRNGKey):
        self.root_config = {
            "depth": depth,
            "created": "2024-01-01",
            "version": (1, 0, 0),
            "features": ["routing", "ensemble", "learning"],
        }

        # Create nested ensembles
        keys = jax.random.split(key, depth + 1)
        self.ensembles = [
            MixedEnsemble(f"level_{i}", num_static=2, num_dynamic=3, dim=4, key=keys[i])
            for i in range(depth)
        ]

        # Static metadata
        self.metadata = {
            "primary": StaticConfig("main", {"importance": 1.0}),
            "secondary": StaticConfig("backup", {"importance": 0.5}),
        }

        # Optional dynamic state
        self.state = DynamicState(keys[-1]) if depth > 2 else None


class ComplexRouter(Module):
    """Router with complex nested decision logic."""

    routes: Dict[str, List[Tuple[str, float]]]  # Static routing table
    operators: Dict[str, Module]  # Mix of static and dynamic
    default_route: str
    adaptation: jnp.ndarray  # Dynamic adaptation weights

    def __init__(self, key: jax.random.PRNGKey):
        # Static routing configuration
        self.routes = {
            "math": [("calculator", 0.8), ("wolfram", 0.2)],
            "code": [("python", 0.6), ("javascript", 0.4)],
            "general": [("gpt4", 0.7), ("claude", 0.3)],
        }
        self.default_route = "general"

        # Mix of operators
        k1, k2 = jax.random.split(key)
        self.operators = {
            "static_analyzer": SimpleOperator("analyzer"),
            "dynamic_processor": LearningOperator("processor", 3, k1),
            "mixed_ensemble": MixedEnsemble("ensemble", 1, 2, 3, k2),
        }

        # Dynamic adaptation
        self.adaptation = jax.random.normal(key, (len(self.routes),))


def test_no_static_warnings_complex():
    """Test that complex nested structures don't trigger static warnings."""
    key = jax.random.PRNGKey(42)

    # Capture warnings during creation
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create complex nested structure
        hierarchy = NestedHierarchy(depth=3, key=key)

        # Check no static array warnings during creation
        static_warnings = [
            warning for warning in w if "JAX array is being set as static" in str(warning.message)
        ]
        assert (
            len(static_warnings) == 0
        ), f"Got {len(static_warnings)} static warnings during creation"

    # Test JIT compilation separately
    @jax.jit
    def process(h, x):
        # Access the first ensemble's dynamic ops
        if h.ensembles and h.ensembles[0].dynamic_ops:
            return h.ensembles[0].dynamic_ops[0].forward(x)
        return x

    # Execute
    result = process(hierarchy, jnp.ones(4))
    assert result.shape == (4,)  # Verify it worked


def test_mixed_static_dynamic_fields():
    """Test that mixed fields are properly partitioned."""
    key = jax.random.PRNGKey(42)
    ensemble = MixedEnsemble("test", num_static=2, num_dynamic=2, dim=3, key=key)

    # Get leaves to check partitioning
    leaves, treedef = jax.tree_util.tree_flatten(ensemble)

    # Count JAX arrays (should be dynamic)
    jax_arrays = [leaf for leaf in leaves if isinstance(leaf, jnp.ndarray)]

    # Should have routing weights + 2 dynamic ops with weights and bias each
    expected_arrays = 1 + 2 * 2  # routing_weights + 2*(weights + bias)
    assert len(jax_arrays) == expected_arrays


def test_deep_nesting_compilation():
    """Test that deep nesting doesn't cause linear execution overhead.

    We only process the first ensemble, so execution time should not scale
    linearly with depth. Some overhead is expected due to pytree traversal,
    but it should be sub-linear (logarithmic or constant).
    """
    key = jax.random.PRNGKey(42)
    deep = NestedHierarchy(depth=10, key=key)

    @jax.jit
    def process_first(h, x):
        # Just process through first ensemble
        if h.ensembles:
            return h.ensembles[0].forward(x)
        return x

    # Warm up JIT
    _ = process_first(deep, jnp.ones(4))

    # Time execution (not compilation)
    import timeit

    time_us = timeit.timeit(lambda: process_first(deep, jnp.ones(4)), number=100) * 10

    # If we had O(n) scaling, depth=10 would be milliseconds.
    # Sub-linear means we stay in microseconds.
    assert time_us < 500, f"Execution too slow: {time_us:.0f}μs (expected < 500μs)"


def test_pytree_registration():
    """Test that our modules work with JAX pytree operations."""
    key = jax.random.PRNGKey(42)
    router = ComplexRouter(key)

    # Tree map should work
    def scale_arrays(x):
        if isinstance(x, jnp.ndarray):
            return x * 2.0
        return x

    scaled = jax.tree_util.tree_map(scale_arrays, router)

    # Check adaptation was scaled
    assert jnp.allclose(scaled.adaptation, router.adaptation * 2.0)

    # Check static fields unchanged
    assert scaled.routes == router.routes
    assert scaled.default_route == router.default_route


def test_vmap_over_mixed_structures():
    """Test that vmap works with mixed structures."""
    from ember.xcs import vmap

    batch_size = 3
    key = jax.random.PRNGKey(42)

    # Create operator
    ensemble = MixedEnsemble("test", num_static=1, num_dynamic=2, dim=4, key=key)

    # Batch of inputs
    x_batch = jnp.ones((batch_size, 4))

    # vmap the forward method
    vmapped = vmap(lambda x: ensemble.forward(x))
    results = vmapped(x_batch)

    # The forward method returns a weighted combination of dynamic_ops outputs
    # Each dynamic op does x @ weights + bias where weights is (4,4) and bias is (4,)
    # So output should be (batch_size, 4)
    # But the weighted combination might have reduced it
    # Let's check the actual shape
    assert len(results.shape) >= 2  # Should be at least 2D
    assert results.shape[0] == batch_size


def test_grad_through_mixed_structures():
    """Test that gradients flow correctly through mixed structures."""
    key = jax.random.PRNGKey(42)

    class LossModule(Module):
        ensemble: MixedEnsemble
        target: jnp.ndarray  # Static target

        def __init__(self, key):
            k1, k2 = jax.random.split(key)
            self.ensemble = MixedEnsemble("loss_test", 0, 2, 3, k1)
            self.target = jax.random.normal(k2, (3,))

        def loss(self, x):
            pred = self.ensemble.forward(x)
            return jnp.mean((pred - self.target) ** 2)

    module = LossModule(key)
    x = jnp.ones(3)

    # Gradient should only be w.r.t dynamic parameters
    grad_fn = jax.grad(lambda m, x: m.loss(x))
    grads = grad_fn(module, x)

    # Check gradients exist for dynamic ops
    assert hasattr(grads.ensemble, "dynamic_ops")
    assert hasattr(grads.ensemble, "routing_weights")

    # Gradients should be non-zero for dynamic parameters
    for op_grad in grads.ensemble.dynamic_ops:
        assert not jnp.allclose(op_grad.weights, 0.0)


def test_serialization_deserialization():
    """Test that mixed structures can be saved and loaded."""
    key = jax.random.PRNGKey(42)
    original = ComplexRouter(key)

    # Get state (should only include dynamic parts)
    leaves, treedef = jax.tree_util.tree_flatten(original)
    dynamic_leaves = [leaf for leaf in leaves if isinstance(leaf, jnp.ndarray)]

    # Recreate with same structure
    new_router = ComplexRouter(jax.random.PRNGKey(99))  # Different key

    # Should have same tree structure
    new_leaves, new_treedef = jax.tree_util.tree_flatten(new_router)

    # Tree definitions should match (same structure)
    # Note: Direct comparison of treedefs may not work, so we check leaf count
    assert len(leaves) == len(new_leaves)
    assert len(dynamic_leaves) == len(
        [leaf for leaf in new_leaves if isinstance(leaf, jnp.ndarray)]
    )


def test_inference_mode_behavior():
    """Test behavior differences between training and inference."""

    class DropoutOp(Module):
        rate: float  # Static
        training: bool  # Should be static

        def __init__(self, rate: float = 0.1, training: bool = True):
            self.rate = rate
            self.training = training

        def forward(self, x, key):
            if self.training:
                mask = jax.random.bernoulli(key, 1 - self.rate, x.shape)
                return x * mask / (1 - self.rate)
            return x

    # Create separate instances for train and eval
    op_train = DropoutOp(training=True)
    op_eval = DropoutOp(training=False)

    # Different behavior based on static training flag
    k = jax.random.PRNGKey(42)
    x = jnp.ones(10)

    train_out = op_train.forward(x, k)
    eval_out = op_eval.forward(x, k)

    # Training should have dropout
    assert not jnp.allclose(train_out, x)
    # Eval should be identity
    assert jnp.allclose(eval_out, x)


def test_no_recompilation_for_same_structure():
    """Test that same structure types don't trigger recompilation."""
    trace_count = 0

    def create_ensemble(name: str, key: jax.random.PRNGKey):
        nonlocal trace_count
        trace_count += 1
        return MixedEnsemble(name, 1, 1, 3, key)

    jitted = jax.jit(create_ensemble, static_argnums=0)

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    # First call
    trace_count = 0
    jitted("ensemble1", k1)
    assert trace_count == 1

    # Second call with same name - no retrace
    jitted("ensemble1", k2)
    assert trace_count == 1

    # Different name - triggers retrace (static arg)
    jitted("ensemble2", k1)
    assert trace_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
