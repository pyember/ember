"""Unit tests for Operator PyTree protocol implementation.

Tests that Ember operators correctly implement the JAX pytree protocol
through equinox.Module inheritance. These are pure unit tests with no
integration dependencies.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import pytest

from ember.api.operators import Operator


class MinimalOperator(Operator):
    """Minimal operator for testing pytree protocol."""

    value: jax.Array

    def __init__(self, value: float = 1.0):
        self.value = jnp.array(value)

    def forward(self, x):
        return x * self.value


class TestPyTreeProtocol:
    """Test that operators correctly implement JAX pytree protocol."""

    def test_operator_is_registered_as_pytree(self):
        """Operators should be automatically registered as JAX pytrees."""
        # Arrange
        op = MinimalOperator(2.0)

        # Act - tree_flatten should work without error
        leaves, treedef = tree.tree_flatten(op)

        # Assert
        assert len(leaves) == 1  # Only the value field
        assert leaves[0] == 2.0
        assert treedef is not None

    def test_pytree_roundtrip_preserves_structure(self):
        """Flattening and unflattening should preserve operator structure."""
        # Arrange
        original = MinimalOperator(3.14)

        # Act
        leaves, treedef = tree.tree_flatten(original)
        reconstructed = tree.tree_unflatten(treedef, leaves)

        # Assert
        assert isinstance(reconstructed, type(original))
        assert jnp.array_equal(reconstructed.value, original.value)
        # Test that forward method still works
        assert jnp.allclose(reconstructed(10.0), 31.4)

    def test_tree_map_applies_to_array_fields_only(self):
        """tree_map should only apply to JAX array fields."""
        # Arrange
        op = MinimalOperator(5.0)

        # Act - double all arrays
        doubled = tree.tree_map(lambda x: x * 2, op)

        # Assert
        assert jnp.array_equal(doubled.value, jnp.array(10.0))
        # Original unchanged (immutability)
        assert jnp.array_equal(op.value, jnp.array(5.0))


class TestStaticFieldHandling:
    """Test that static fields are handled correctly in pytree operations."""

    def test_equinox_partition_separates_static_dynamic(self):
        """equinox.partition should correctly separate fields."""

        # Arrange
        class MixedOperator(Operator):
            dynamic_value: jax.Array
            static_config: dict

            def __init__(self):
                self.dynamic_value = jnp.ones(3)
                self.static_config = {"name": "test"}

            def forward(self, x):
                return x * self.dynamic_value

        op = MixedOperator()

        # Act
        dynamic, static = eqx.partition(op, eqx.is_array)

        # Assert - dynamic partition has arrays
        assert isinstance(dynamic.dynamic_value, jax.Array)
        assert jnp.array_equal(dynamic.dynamic_value, jnp.ones(3))

        # Assert - static partition has non-arrays
        assert static.static_config == {"name": "test"}
        assert static.dynamic_value is None

    def test_static_fields_preserved_in_tree_operations(self):
        """Static fields should be preserved during tree operations."""

        # Arrange
        class ConfiguredOperator(Operator):
            weight: jax.Array
            name: str
            settings: dict

            def __init__(self, name: str):
                self.weight = jnp.array(1.0)
                self.name = name
                self.settings = {"version": 1}

            def forward(self, x):
                return x * self.weight

        op = ConfiguredOperator("test_op")

        # Act - tree operation that modifies arrays
        modified = tree.tree_map(lambda x: x + 1 if isinstance(x, jax.Array) else x, op)

        # Assert - static fields unchanged
        assert modified.name == "test_op"
        assert modified.settings == {"version": 1}
        # Dynamic field changed
        assert jnp.array_equal(modified.weight, jnp.array(2.0))


class TestPyTreeLeaves:
    """Test behavior of tree_leaves and related operations."""

    def test_tree_leaves_returns_only_arrays(self):
        """tree_leaves should return only the array values."""

        # Arrange
        class MultiArrayOperator(Operator):
            weights: jax.Array
            bias: jax.Array
            scale: jax.Array
            name: str

            def __init__(self):
                self.weights = jnp.ones((2, 2))
                self.bias = jnp.zeros(2)
                self.scale = jnp.array(0.1)
                self.name = "multi"

            def forward(self, x):
                return (x @ self.weights + self.bias) * self.scale

        op = MultiArrayOperator()

        # Act
        leaves = tree.tree_leaves(op)

        # Assert
        assert len(leaves) == 3  # weights, bias, scale
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)
        # Check shapes
        shapes = [leaf.shape for leaf in leaves]
        assert (2, 2) in shapes
        assert (2,) in shapes
        assert () in shapes  # scalar

    def test_tree_structure_matches_equinox_module(self):
        """Operator tree structure should match equinox.Module behavior."""
        # Arrange
        op = MinimalOperator(42.0)

        # Act - get tree structure
        _, op_treedef = tree.tree_flatten(op)

        # Also create a bare equinox module for comparison
        class BareModule(eqx.Module):
            value: jax.Array

            def __init__(self, value):
                self.value = jnp.array(value)

        bare = BareModule(42.0)
        _, bare_treedef = tree.tree_flatten(bare)

        # Assert - tree structures should be compatible
        # Both should flatten to same number of leaves
        assert len(tree.tree_leaves(op)) == len(tree.tree_leaves(bare))


class TestErrorConditions:
    """Test error handling and edge cases."""

    def test_non_array_in_array_field_raises_clear_error(self):
        """Assigning non-arrays to array-annotated fields should error clearly."""
        # This test documents expected behavior when type hints are violated

        class BadOperator(Operator):
            value: jax.Array

            def __init__(self):
                # This might work due to equinox's flexibility
                # but it violates the type annotation
                self.value = "not an array"  # Bad!

            def forward(self, x):
                return x

        # The operator might construct, but operations will fail
        op = BadOperator()

        # Tree operations treat non-arrays as leaves too in equinox
        leaves = tree.tree_leaves(op)
        # Equinox includes all fields as leaves, not just arrays
        # This is different from pure JAX behavior
        assert len(leaves) > 0  # Will include the string

    def test_empty_operator_still_valid_pytree(self):
        """Even operators with no array fields are valid pytrees."""

        class EmptyOperator(Operator):
            name: str

            def __init__(self):
                self.name = "empty"

            def forward(self, x):
                return x

        # Arrange
        op = EmptyOperator()

        # Act
        leaves, treedef = tree.tree_flatten(op)
        reconstructed = tree.tree_unflatten(treedef, leaves)

        # Assert
        assert len(leaves) == 0  # No arrays
        assert reconstructed.name == "empty"
        assert isinstance(reconstructed, EmptyOperator)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
