"""Test that Ember operators are fully compatible JAX pytrees.

This test module demonstrates the key architectural insight: Ember operators
inherit from equinox.Module, making them automatically compatible with all
JAX transformations without any IR translation.

Test Structure:
- Focused on single concept: operators as JAX pytrees
- Clear demonstration of static vs dynamic field handling
- Practical examples showing real-world usage patterns
"""

from typing import Any, Dict, List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as tree

from ember.api.operators import Operator


class MixedFieldOperator(Operator):
    """Operator with mix of static and dynamic fields for testing."""

    # Dynamic fields (JAX arrays)
    weights: jax.Array
    bias: jax.Array
    embeddings: jax.Array

    # Static fields - must be explicitly marked for containers
    model_name: str  # Auto-marked static by EmberModuleMeta (primitive type)
    config: Dict[str, Any] = eqx.field(static=True)
    routes: List[str] = eqx.field(static=True)
    temperature_constant: float  # Auto-marked static by EmberModuleMeta

    def __init__(self, dim: int = 3):
        # Dynamic - will participate in JAX transformations
        self.weights = jnp.ones((dim, dim))
        self.bias = jnp.zeros(dim)
        self.embeddings = jax.random.normal(jax.random.PRNGKey(0), (4, dim))

        # Static - compile-time constants
        self.model_name = "gpt-4"
        self.config = {"learning_rate": 0.01, "batch_size": 32}
        self.routes = ["analytical", "creative", "coding", "simple"]
        self.temperature_constant = 1.0

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Simple forward pass mixing static and dynamic operations."""
        # Use dynamic fields
        hidden = jnp.dot(x, self.weights) + self.bias

        # Use static field in computation (but it's still static!)
        scaled = hidden / self.temperature_constant

        # Route based on static config
        if self.config.get("use_embeddings", True):
            # Mix with embeddings
            scores = jnp.dot(self.embeddings, scaled)
            return jax.nn.softmax(scores)
        else:
            return jax.nn.relu(scaled)


class TestOperatorAsPyTree:
    """Test operators work as JAX pytrees without any special handling."""

    def test_operators_are_pytrees(self):
        """Operators are automatically registered as JAX pytrees via equinox."""
        op = MixedFieldOperator(dim=3)

        # Should be able to use all pytree operations
        leaves = tree.tree_leaves(op)
        assert len(leaves) > 0

        # Should be able to flatten and unflatten
        flat, treedef = tree.tree_flatten(op)
        reconstructed = tree.tree_unflatten(treedef, flat)

        # Verify reconstruction preserves all fields
        assert jnp.allclose(reconstructed.weights, op.weights)
        assert reconstructed.model_name == op.model_name
        assert reconstructed.config == op.config

    def test_static_dynamic_partitioning(self):
        """Equinox correctly partitions static and dynamic fields."""
        op = MixedFieldOperator(dim=3)

        # Partition into dynamic (arrays) and static (everything else)
        dynamic, static = eqx.partition(op, eqx.is_array)

        # Check dynamic fields are arrays
        assert isinstance(dynamic.weights, jax.Array)
        assert isinstance(dynamic.bias, jax.Array)
        assert isinstance(dynamic.embeddings, jax.Array)

        # Check static fields in the static partition
        assert static.model_name == "gpt-4"
        assert static.config == {"learning_rate": 0.01, "batch_size": 32}
        assert static.routes == ["analytical", "creative", "coding", "simple"]

        # Arrays should be None in static partition
        assert static.weights is None
        assert static.bias is None

        # The key insight: equinox.partition separates the pytree into two parts
        # - dynamic: has arrays, everything else is the same structure
        # - static: has non-arrays, arrays are None

    def test_jax_grad_only_dynamic_fields(self):
        """JAX grad automatically computes gradients only for dynamic fields."""
        op = MixedFieldOperator(dim=3)

        def loss_fn(operator, x):
            output = operator(x)
            return jnp.sum(output**2)

        x = jnp.ones(3)

        # Compute gradients - should only get gradients for JAX arrays
        grads = jax.grad(loss_fn)(op, x)

        # Dynamic fields should have gradients
        assert hasattr(grads, "weights") and grads.weights is not None
        assert hasattr(grads, "bias") and grads.bias is not None
        assert hasattr(grads, "embeddings") and grads.embeddings is not None

        # Static fields should still exist but not participate in gradients
        # They maintain their values (not None) because they're static
        assert grads.model_name == "gpt-4"
        assert grads.config == {"learning_rate": 0.01, "batch_size": 32}
        assert grads.routes == ["analytical", "creative", "coding", "simple"]
        assert grads.temperature_constant == 1.0

    def test_jax_vmap_shares_static_fields(self):
        """JAX vmap batches dynamic fields while sharing static fields."""
        op = MixedFieldOperator(dim=3)

        # Create batch of inputs
        batch_x = jnp.ones((5, 3))

        # vmap the operator's forward method
        vmapped_forward = jax.vmap(op.forward)
        outputs = vmapped_forward(batch_x)

        assert outputs.shape == (5, 4)  # Batch of 5, 4 routes

        # More interestingly, vmap the whole operator
        def process_with_op(x, weights_scale):
            # Create modified operator (simulating different parameters)
            scaled_op = op.update_params(weights=op.weights * weights_scale)
            return scaled_op(x)

        # Vmap over different weight scales
        x = jnp.ones(3)
        weight_scales = jnp.array([0.5, 1.0, 1.5, 2.0])
        vmapped_process = jax.vmap(process_with_op, in_axes=(None, 0))

        results = vmapped_process(x, weight_scales)
        assert results.shape == (4, 4)  # 4 different scales, 4 routes

    def test_jax_jit_static_fields_as_constants(self):
        """JAX jit compiles static fields as constants."""
        op = MixedFieldOperator(dim=3)

        @jax.jit
        def jitted_forward(operator, x):
            # Static fields are compiled as constants
            # This branching happens at compile time!
            if operator.config["batch_size"] == 32:
                scale = 2.0
            else:
                scale = 1.0

            return operator(x) * scale

        x = jnp.ones(3)

        # First call triggers compilation
        result1 = jitted_forward(op, x)

        # Changing dynamic fields works fine
        op2 = op.update_params(weights=op.weights * 2)
        result2 = jitted_forward(op2, x)

        assert not jnp.allclose(result1, result2)

        # But changing static fields would trigger recompilation
        # (This is good - static fields become compile-time constants!)
        # Note: update_params on static fields may not work as expected
        # because they're marked static at class definition time

    def test_tree_map_on_operators(self):
        """Standard tree_map operations work on operators."""
        op = MixedFieldOperator(dim=3)

        # Scale all arrays by 2
        def scale_if_array(x):
            if isinstance(x, jax.Array):
                return x * 2
            return x

        scaled_op = tree.tree_map(scale_if_array, op)

        # Dynamic fields should be scaled
        assert jnp.allclose(scaled_op.weights, op.weights * 2)
        assert jnp.allclose(scaled_op.bias, op.bias * 2)

        # Static fields unchanged
        assert scaled_op.model_name == op.model_name
        assert scaled_op.config == op.config

    def test_pytree_in_pytree(self):
        """Operators can be nested in other pytree structures."""
        op1 = MixedFieldOperator(dim=3)
        op2 = MixedFieldOperator(dim=3)

        # Put operators in standard Python containers
        op_dict = {"encoder": op1, "decoder": op2}
        op_list = [op1, op2]
        op_tuple = (op1, op2)

        # All should work with tree operations
        for container in [op_dict, op_list, op_tuple]:
            leaves = tree.tree_leaves(container)
            assert len(leaves) > 0

            # Should be able to map over the whole structure
            # Map over just the leaves instead
            def zero_if_array(x):
                if isinstance(x, jax.Array):
                    return jnp.zeros_like(x)
                return x

            zeroed = tree.tree_map(zero_if_array, container)

            # Check that array leaves were zeroed
            zeroed_leaves = tree.tree_leaves(zeroed)
            array_leaves = [leaf for leaf in zeroed_leaves if isinstance(leaf, jax.Array)]
            assert all(jnp.allclose(leaf, 0) for leaf in array_leaves)

    def test_no_ir_translation_needed(self):
        """Demonstrate that operators work directly with JAX - no IR needed."""
        op = MixedFieldOperator(dim=3)

        # All of these "just work" without any XCS IR translation

        # 1. Direct JAX transformations
        # Need to wrap in a function to avoid hashing issues
        @jax.jit
        def jitted_forward(x):
            return op.forward(x)

        grad_op = jax.grad(lambda x: jnp.sum(op(x)))
        vmapped_op = jax.vmap(op.forward)

        x = jnp.ones(3)
        batch_x = jnp.ones((4, 3))

        # All work directly
        jitted_result = jitted_forward(x)
        grads = grad_op(x)
        batch_results = vmapped_op(batch_x)

        assert jitted_result.shape == (4,)
        assert grads.shape == (3,)
        assert batch_results.shape == (4, 4)

        # 2. Composed transformations
        jitted_grad = jax.jit(jax.grad(lambda op, x: jnp.sum(op(x))))
        op_grads = jitted_grad(op, x)

        assert op_grads.weights is not None
        assert op_grads.model_name == "gpt-4"  # Static field preserved

    def test_functional_update_pattern(self):
        """The update_params pattern maintains pytree compatibility."""
        op = MixedFieldOperator(dim=3)

        # Define training step that works with JAX transformations
        @jax.jit
        def train_step(operator, x, learning_rate):
            # Forward pass
            def loss_fn(op):
                return jnp.sum(op(x) ** 2)

            # Compute gradients (only for dynamic fields!)
            grads = jax.grad(loss_fn)(operator)

            # Functional update using tree_map
            def update_param(param, grad):
                if grad is not None:  # Only update dynamic fields
                    return param - learning_rate * grad
                return param

            # This works because operators are pytrees!
            return tree.tree_map(update_param, operator, grads)

        x = jnp.ones(3)
        new_op = train_step(op, x, 0.01)

        # Verify updates
        assert not jnp.allclose(new_op.weights, op.weights)
        assert new_op.model_name == op.model_name  # Static unchanged


if __name__ == "__main__":
    # Run key tests to demonstrate the concept
    test = TestOperatorAsPyTree()

    print("Testing operators as JAX pytrees...")
    test.test_operators_are_pytrees()
    print("✓ Operators are automatically JAX pytrees")

    test.test_static_dynamic_partitioning()
    print("✓ Static/dynamic fields partition correctly")

    test.test_jax_grad_only_dynamic_fields()
    print("✓ Gradients computed only for dynamic fields")

    test.test_jax_vmap_shares_static_fields()
    print("✓ vmap batches dynamic, shares static fields")

    test.test_jax_jit_static_fields_as_constants()
    print("✓ JIT compiles static fields as constants")

    test.test_no_ir_translation_needed()
    print("✓ No IR translation needed - JAX works directly!")

    print("\n✅ All tests passed! Operators are fully compatible JAX pytrees.")
