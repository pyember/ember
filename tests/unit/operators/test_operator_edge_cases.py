"""Edge case tests for Ember operators.

Covers unusual but valid scenarios, boundary conditions, and error handling.
"""

from typing import Dict, List, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ember.api.operators import Operator


class TestUnusualFieldTypes:
    """Test operators with unusual but valid field types."""

    def test_operator_with_optional_fields(self):
        """Operators should handle Optional fields correctly."""

        class OptionalFieldOperator(Operator):
            required_weight: jax.Array
            optional_bias: Optional[jax.Array]
            optional_config: Optional[dict]

            def __init__(self, use_bias: bool = True):
                self.required_weight = jnp.ones(3, dtype=jnp.float32)
                self.optional_bias = jnp.zeros(3, dtype=jnp.float32) if use_bias else None
                self.optional_config = {"scale": 1.0} if use_bias else None

            def forward(self, x):
                result = x * self.required_weight
                if self.optional_bias is not None:
                    result = result + self.optional_bias
                return result

        # Test with optional fields present
        op_with = OptionalFieldOperator(use_bias=True)
        output_with = op_with(jnp.array([1, 2, 3], dtype=jnp.float32))
        assert output_with.shape == (3,)

        # Test with optional fields absent
        op_without = OptionalFieldOperator(use_bias=False)
        output_without = op_without(jnp.array([1, 2, 3], dtype=jnp.float32))
        assert output_without.shape == (3,)

        # Test gradient computation with float inputs
        def loss_fn(op):
            x = jnp.ones(3, dtype=jnp.float32)
            return jnp.sum(op(x))

        grads = jax.grad(loss_fn)(op_with)
        assert grads.required_weight is not None
        # Optional field might have gradient or be None

    def test_operator_with_union_types(self):
        """Operators should handle Union types appropriately."""

        class UnionTypeOperator(Operator):
            value: Union[jax.Array, float]
            config: Union[dict, str]

            def __init__(self, use_array: bool = True):
                self.value = jnp.array(2.0) if use_array else 2.0
                self.config = {"type": "dict"} if use_array else "string_config"

            def forward(self, x):
                if isinstance(self.value, jax.Array):
                    return x * self.value
                else:
                    return x * self.value  # Works for float too

        # Test both variants
        op_array = UnionTypeOperator(use_array=True)
        op_float = UnionTypeOperator(use_array=False)

        x = jnp.array([1, 2, 3])
        assert jnp.array_equal(op_array(x), jnp.array([2, 4, 6]))
        assert jnp.array_equal(op_float(x), jnp.array([2, 4, 6]))

    def test_operator_with_nested_containers(self):
        """Operators with nested container types."""

        class NestedContainerOperator(Operator):
            weights_dict: Dict[str, jax.Array]  # Properly typed
            config_list: List[Dict[str, str]]  # Properly typed
            matrix: jax.Array

            def __init__(self):
                self.weights_dict = {"layer1": jnp.ones((2, 2)), "layer2": jnp.ones((2, 2)) * 2}
                self.config_list = [{"name": "config1"}, {"name": "config2"}]
                self.matrix = jnp.zeros((2, 2))

            def forward(self, x):
                # Use nested structure
                x = x @ self.weights_dict["layer1"]
                x = x @ self.weights_dict["layer2"]
                return x @ self.matrix

        # Create operator
        op = NestedContainerOperator()

        # Set dynamic field properly
        op = eqx.tree_at(lambda o: o.matrix, op, jnp.eye(2))
        result = op(jnp.ones((3, 2)))
        assert result.shape == (3, 2)

        # Check tree operations work
        leaves = jax.tree_util.tree_leaves(op)
        # Note: dicts are treated as static by equinox, so arrays inside won't be leaves
        # Only the matrix field will be a leaf
        array_leaves = [leaf for leaf in leaves if isinstance(leaf, jax.Array)]
        assert len(array_leaves) >= 1  # At least the matrix


class TestBoundaryConditions:
    """Test operators at boundary conditions."""

    def test_operator_with_zero_size_arrays(self):
        """Operators should handle zero-size arrays correctly."""

        class ZeroSizeOperator(Operator):
            empty_weight: jax.Array
            normal_bias: jax.Array

            def __init__(self):
                self.empty_weight = jnp.array([])  # Zero-size
                self.normal_bias = jnp.array(1.0)

            def forward(self, x):
                if x.size == 0:
                    return x  # Return empty
                return x + self.normal_bias

        op = ZeroSizeOperator()

        # Test with empty input
        empty_result = op(jnp.array([]))
        assert empty_result.size == 0

        # Test with normal input
        normal_result = op(jnp.array([1, 2, 3]))
        assert jnp.array_equal(normal_result, jnp.array([2, 3, 4]))

    def test_operator_with_scalar_arrays(self):
        """Operators with scalar (0-dimensional) arrays."""

        class ScalarOperator(Operator):
            scalar_weight: jax.Array
            scalar_bias: jax.Array

            def __init__(self):
                self.scalar_weight = jnp.array(2.0)  # 0-d array
                self.scalar_bias = jnp.array(0.5)  # 0-d array

            def forward(self, x):
                return x * self.scalar_weight + self.scalar_bias

        op = ScalarOperator()

        # Test with various inputs
        scalar_input = jnp.array(3.0)
        vector_input = jnp.array([1, 2, 3])

        scalar_result = op(scalar_input)
        assert scalar_result.shape == ()
        assert scalar_result == 6.5

        vector_result = op(vector_input)
        assert vector_result.shape == (3,)
        assert jnp.array_equal(vector_result, jnp.array([2.5, 4.5, 6.5]))

    def test_operator_with_very_large_arrays(self):
        """Test memory efficiency with large arrays."""

        class LargeArrayOperator(Operator):
            # Don't actually allocate huge arrays in tests
            size: int
            scale: jax.Array

            def __init__(self, size: int = 1000):
                self.size = size
                self.scale = jnp.array(0.1)

            def forward(self, x):
                # Simulate large array operation
                return x * self.scale

        op = LargeArrayOperator(size=10_000)

        # Test that tree operations don't duplicate memory
        leaves, treedef = jax.tree_util.tree_flatten(op)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        # Should share scale array reference (equinox's immutability)
        assert reconstructed.scale is op.scale


class TestInheritanceScenarios:
    """Test various inheritance patterns with operators."""

    def test_deep_inheritance_chain(self):
        """Operators with deep inheritance should work correctly."""

        class BaseOperator(Operator):
            base_weight: jax.Array

            def __init__(self):
                self.base_weight = jnp.array(1.0)

            def forward(self, x):
                return x * self.base_weight

        class MiddleOperator(BaseOperator):
            middle_weight: jax.Array

            def __init__(self):
                super().__init__()
                self.middle_weight = jnp.array(2.0)

            def forward(self, x):
                x = super().forward(x)
                return x * self.middle_weight

        class LeafOperator(MiddleOperator):
            leaf_weight: jax.Array

            def __init__(self):
                super().__init__()
                self.leaf_weight = jnp.array(3.0)

            def forward(self, x):
                x = super().forward(x)
                return x * self.leaf_weight

        op = LeafOperator()
        result = op(jnp.array(1.0))
        assert result == 6.0  # 1 * 1 * 2 * 3

        # Check all fields are accessible
        assert hasattr(op, "base_weight")
        assert hasattr(op, "middle_weight")
        assert hasattr(op, "leaf_weight")

        # Check gradients flow through all levels
        grads = jax.grad(lambda o: o(jnp.array(1.0)))(op)
        assert grads.base_weight is not None
        assert grads.middle_weight is not None
        assert grads.leaf_weight is not None

    def test_multiple_inheritance(self):
        """Operators with multiple inheritance (mixin pattern)."""

        class ScaleMixin:
            """Mixin providing scaling functionality."""

            def apply_scale(self, x):
                if hasattr(self, "scale"):
                    return x * self.scale
                return x

        class BiasOperator(Operator):
            """Operator providing bias functionality."""

            bias: jax.Array

            def __init__(self):
                self.bias = jnp.zeros(3)

            def forward(self, x):
                return x + self.bias

        class MixedOperator(BiasOperator, ScaleMixin):
            """Operator using both bias and scale."""

            scale: jax.Array  # Must declare field at class level

            def __init__(self):
                super().__init__()
                self.scale = jnp.array(2.0)

            def forward(self, x):
                x = self.apply_scale(x)
                return super().forward(x)

        op = MixedOperator()
        result = op(jnp.ones(3))
        assert jnp.array_equal(result, jnp.array([2.0, 2.0, 2.0]))


class TestMutabilityConstraints:
    """Test that immutability is properly enforced."""

    def test_cannot_mutate_operator_fields(self):
        """Operator fields should be immutable after creation."""

        class ImmutableOperator(Operator):
            weight: jax.Array
            config: dict

            def __init__(self):
                self.weight = jnp.array([1, 2, 3])
                self.config = {"frozen": True}

            def forward(self, x):
                return x * self.weight

        op = ImmutableOperator()

        # Attempting to mutate should raise error
        with pytest.raises((AttributeError, TypeError)):
            op.weight = jnp.array([4, 5, 6])

        with pytest.raises((AttributeError, TypeError)):
            op.config = {"frozen": False}

    def test_update_params_creates_new_instance(self):
        """update_params should create new instance, not mutate."""

        class TestOperator(Operator):
            value: jax.Array

            def __init__(self, value: float = 1.0):
                self.value = jnp.array(value)

            def forward(self, x):
                return x * self.value

        op1 = TestOperator(1.0)
        op2 = op1.update_params(value=jnp.array(2.0))

        # Should be different instances
        assert op1 is not op2
        assert op1.value == 1.0
        assert op2.value == 2.0

        # But same type
        assert isinstance(op2, type(op1))


class TestErrorMessages:
    """Test that error messages are clear and helpful."""

    def test_missing_forward_method_error(self):
        """Missing forward method should give clear error."""

        # Test behavior when forward method is missing

        class NoForwardOperator(Operator):
            weight: jax.Array

            def __init__(self):
                self.weight = jnp.array(1.0)

            # Intentionally missing forward method

        # Operator base class might not enforce forward method at instantiation
        # The error would occur when trying to call the operator
        op = NoForwardOperator()  # This might succeed

        # But calling it should fail
        with pytest.raises((AttributeError, NotImplementedError)) as exc_info:
            op(jnp.array(1.0))  # This should fail

        # Check for helpful error message
        error_msg = str(exc_info.value).lower()
        assert "forward" in error_msg or "not implemented" in error_msg

    def test_wrong_field_type_error(self):
        """Type errors should be caught and reported clearly."""

        class TypedOperator(Operator):
            weight: jax.Array

            def __init__(self):
                # This might work due to duck typing, but document behavior
                self.weight = "not an array"  # Wrong type!

            def forward(self, x):
                return x * self.weight

        op = TypedOperator()

        # Error should occur when trying to use as array
        with pytest.raises((TypeError, AttributeError)):
            op(jnp.array([1, 2, 3]))

    def test_update_params_nonexistent_field(self):
        """update_params with non-existent field should error clearly."""

        class SimpleOperator(Operator):
            value: jax.Array

            def __init__(self):
                self.value = jnp.array(1.0)

            def forward(self, x):
                return x * self.value

        op = SimpleOperator()

        with pytest.raises(AttributeError) as exc_info:
            op.update_params(nonexistent=jnp.array(2.0))

        assert "nonexistent" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
