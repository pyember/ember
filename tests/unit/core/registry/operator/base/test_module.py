"""Unit tests for EmberModule functionality.

This module tests the core functionality of the EmberModule system, including:
1. Field creation and metadata handling
2. Metaclass behavior for automatic dataclass decoration and registration 
3. Tree flattening and unflattening for transformation operations
4. EmberModule initialization and immutability
5. EmberModule integration with the transformation system

These tests ensure that the EmberModule system provides a robust foundation
for immutable, tree-transformable modules as used in the Ember framework.
"""

from __future__ import annotations

import dataclasses
import inspect
import threading
import sys
from typing import Any, List, Tuple

import pytest

from ember.core.registry.operator.base._module import (
    EmberModule,
    BoundMethod,
    static_field,
    ember_field,
    _make_initable_wrapper,
    _flatten_ember_module,
    _unflatten_ember_module,
    _thread_local,
    EmberModuleMeta,
)
from ember.core.registry.operator.exceptions import (
    BoundMethodNotInitializedError,
    FlattenError,
)
from ember.xcs.utils.tree_util import (
    tree_flatten,
    tree_unflatten,
    _pytree_registry,
)


# -----------------------------------------------------------------------------
# Field Creation Tests
# -----------------------------------------------------------------------------


def test_static_field_metadata() -> None:
    """Tests that static_field correctly sets metadata with static=True.

    The static_field function should create a dataclass field with the metadata
    attribute containing {"static": True}, indicating that the field should be
    excluded from tree transformations.
    """
    field_obj = static_field(default=42)
    assert field_obj.default == 42, "Default value not correctly set"
    assert (
        field_obj.metadata.get("static") is True
    ), "static_field should set metadata['static'] to True"

    # Test with additional kwargs
    field_obj_with_kwargs = static_field(default=42, repr=False)
    assert (
        field_obj_with_kwargs.default == 42
    ), "Default value not correctly set with kwargs"
    assert (
        field_obj_with_kwargs.metadata.get("static") is True
    ), "static flag not set in metadata with kwargs"
    assert field_obj_with_kwargs.repr is False, "repr flag not correctly set"


def test_ember_field_with_converter() -> None:
    """Tests that ember_field correctly sets a converter in metadata.

    When a converter function is provided to ember_field, it should be stored
    in the field's metadata for later use during instance initialization.
    """

    def converter(x: int) -> str:
        return str(x)

    field_obj = ember_field(converter=converter)
    assert (
        field_obj.metadata.get("converter") is converter
    ), "ember_field should store the converter in metadata"


def test_ember_field_static() -> None:
    """Tests that ember_field correctly sets static=True when requested.

    When static=True is passed to ember_field, it should be stored in the field's
    metadata, indicating that the field should be excluded from tree transformations.
    """
    field_obj = ember_field(static=True)
    assert (
        field_obj.metadata.get("static") is True
    ), "ember_field should set metadata['static'] to True when static=True"

    # Test combining static and converter
    def converter(x: int) -> str:
        return str(x)

    field_obj_combined = ember_field(static=True, converter=converter)
    assert (
        field_obj_combined.metadata.get("static") is True
    ), "static flag not set with converter"
    assert (
        field_obj_combined.metadata.get("converter") is converter
    ), "converter not set with static flag"


def test_ember_field_default_factory() -> None:
    """Tests that ember_field correctly handles default_factory.

    ember_field should support specifying a default_factory function that will be
    called to create the default value for a field when not provided at initialization.
    """

    def default_factory() -> List[int]:
        return [1, 2, 3]

    field_obj = ember_field(default_factory=default_factory)
    assert (
        field_obj.default_factory is default_factory
    ), "default_factory not correctly set"


def test_ember_field_all_params() -> None:
    """Tests ember_field with all possible parameters combined.

    ember_field should correctly handle all combinations of parameters.
    """

    def converter(x: int) -> str:
        return str(x)

    def default_factory() -> List[int]:
        return [1, 2, 3]

    # Test with all parameters
    field_obj = ember_field(
        converter=converter,
        static=True,
        default_factory=default_factory,
        init=False,
        repr=False,
    )

    assert field_obj.metadata.get("converter") is converter, "converter not set"
    assert field_obj.metadata.get("static") is True, "static flag not set"
    assert field_obj.default_factory is default_factory, "default_factory not set"
    assert field_obj.init is False, "init flag not set"
    assert field_obj.repr is False, "repr flag not set"


def test_static_field_metadata_merging() -> None:
    """Tests that static_field correctly merges with existing metadata.

    If metadata is provided to static_field, it should be merged with the
    static flag rather than overwriting it.
    """
    existing_metadata = {"existing": "value"}

    field_obj = static_field(metadata=existing_metadata)

    # Both the static flag and existing metadata should be present
    assert field_obj.metadata.get("static") is True, "static flag not set"
    assert field_obj.metadata.get("existing") == "value", "existing metadata lost"


# -----------------------------------------------------------------------------
# Module Registration Tests
# -----------------------------------------------------------------------------


def test_module_registration() -> None:
    """Tests that EmberModule subclasses are registered with the tree system.

    When a new EmberModule subclass is created, it should be automatically
    registered with the tree transformation system via _pytree_registry.
    """

    class NewTestModule(EmberModule):
        pass

    assert (
        NewTestModule in _pytree_registry
    ), "EmberModule subclass should be registered with the tree system"

    # Check that the registration includes flatten and unflatten functions
    flatten_func, unflatten_func = _pytree_registry[NewTestModule]
    assert callable(flatten_func), "Registered flatten function should be callable"
    assert callable(unflatten_func), "Registered unflatten function should be callable"


def test_registration_idempotence() -> None:
    """Tests that registering a class multiple times is handled correctly.

    The tree registration process should handle (or prevent) registering
    the same class multiple times.
    """

    # Create a class to test with
    class IdempotentModule(EmberModule):
        pass

    # The class should already be registered by its creation
    assert IdempotentModule in _pytree_registry

    # Get the originally registered functions
    original_flatten, original_unflatten = _pytree_registry[IdempotentModule]

    # Import the register_tree function directly
    from ember.xcs.utils.tree_util import register_tree

    # Try to register again with dummy functions
    def dummy_flatten(x: Any) -> Tuple[List[Any], Any]:
        return [], None

    def dummy_unflatten(aux: Any, children: List[Any]) -> Any:
        return None

    # This should raise a ValueError
    with pytest.raises(ValueError, match="already registered"):
        register_tree(
            cls=IdempotentModule,
            flatten_func=dummy_flatten,
            unflatten_func=dummy_unflatten,
        )

    # The original registration should still be in place
    current_flatten, current_unflatten = _pytree_registry[IdempotentModule]
    assert current_flatten is original_flatten, "Original flatten function was replaced"
    assert (
        current_unflatten is original_unflatten
    ), "Original unflatten function was replaced"


# -----------------------------------------------------------------------------
# EmberModule Class Tests
# -----------------------------------------------------------------------------


def test_init_method_existence() -> None:
    """Tests that EmberModule has a __init__ method.

    EmberModule should define an __init__ method that can be inspected.
    """
    assert hasattr(
        EmberModule, "__init__"
    ), "EmberModule should have an __init__ method"
    assert callable(EmberModule.__init__), "__init__ should be callable"


def test_init_field_method() -> None:
    """Tests the _init_field method exists.

    The _init_field method should exist on EmberModule.
    """
    assert hasattr(
        EmberModule, "_init_field"
    ), "EmberModule should have _init_field method"
    assert callable(EmberModule._init_field), "_init_field should be callable"


def test_special_methods_exist() -> None:
    """Tests that EmberModule has the expected special methods.

    EmberModule should have __hash__, __eq__, and __repr__ methods.
    """
    assert hasattr(EmberModule, "__hash__"), "EmberModule should have __hash__ method"
    assert callable(EmberModule.__hash__), "__hash__ should be callable"

    assert hasattr(EmberModule, "__eq__"), "EmberModule should have __eq__ method"
    assert callable(EmberModule.__eq__), "__eq__ should be callable"

    assert hasattr(EmberModule, "__repr__"), "EmberModule should have __repr__ method"
    assert callable(EmberModule.__repr__), "__repr__ should be callable"


def test_embermodule_metaclass() -> None:
    """Tests that EmberModule uses EmberModuleMeta as its metaclass.

    The EmberModule class should be created using the EmberModuleMeta metaclass.
    """
    assert isinstance(
        EmberModule, EmberModuleMeta
    ), "EmberModule should use EmberModuleMeta as its metaclass"


def test_dataclass_recognition() -> None:
    """Tests that EmberModule is recognized as a dataclass.

    EmberModule should be processed by the dataclass decorator and recognized
    as a dataclass by the dataclasses.is_dataclass function.
    """
    assert dataclasses.is_dataclass(
        EmberModule
    ), "EmberModule should be recognized as a dataclass"

    # Check that subclasses are also recognized as dataclasses
    class TestModule(EmberModule):
        pass

    assert dataclasses.is_dataclass(
        TestModule
    ), "EmberModule subclass should be recognized as a dataclass"


def test_make_initable_wrapper() -> None:
    """Tests the _make_initable_wrapper function.

    _make_initable_wrapper should create a mutable wrapper class around a frozen class.
    """

    @dataclasses.dataclass(frozen=True)
    class FrozenTestClass:
        x: int = 0

    # Create the wrapper
    MutableWrapper = _make_initable_wrapper(FrozenTestClass)

    # The wrapper should have the same name, qualname, and module as the original
    assert MutableWrapper.__name__ == FrozenTestClass.__name__
    assert MutableWrapper.__qualname__ == FrozenTestClass.__qualname__
    assert MutableWrapper.__module__ == FrozenTestClass.__module__

    # Verify it's a subclass
    assert issubclass(
        MutableWrapper, FrozenTestClass
    ), "Wrapper should be a subclass of the original"

    # Verify it has a __setattr__ method
    assert hasattr(
        MutableWrapper, "__setattr__"
    ), "Wrapper should have a __setattr__ method"


# -----------------------------------------------------------------------------
# BoundMethod Tests (Using Direct Function Access)
# -----------------------------------------------------------------------------


def test_boundmethod_structure() -> None:
    """Tests that the BoundMethod class has the expected structure.

    BoundMethod should be a subclass of EmberModule and have the expected attributes.
    """
    # Check inheritance and attributes
    assert issubclass(
        BoundMethod, EmberModule
    ), "BoundMethod should be a subclass of EmberModule"

    # Check for field definitions
    assert hasattr(
        BoundMethod, "__func__"
    ), "BoundMethod should have a __func__ attribute"
    assert hasattr(
        BoundMethod, "__self__"
    ), "BoundMethod should have a __self__ attribute"
    assert hasattr(BoundMethod, "__call__"), "BoundMethod should have a __call__ method"


def test_boundmethod_call_specification() -> None:
    """Tests that the BoundMethod.__call__ method has the expected specification.

    The __call__ method should accept *args and **kwargs for forwarding to the bound function.
    """
    # Inspect the __call__ method
    signature = inspect.signature(BoundMethod.__call__)

    # Check the parameter names
    param_names = list(signature.parameters.keys())
    assert "self" in param_names, "__call__ should have a 'self' parameter"
    assert "*args" in str(
        signature
    ), "__call__ should accept variable positional arguments"
    assert "**kwargs" in str(
        signature
    ), "__call__ should accept variable keyword arguments"


# -----------------------------------------------------------------------------
# Pytree Integration Tests
# -----------------------------------------------------------------------------


def test_embermodule_pytree_methods() -> None:
    """Tests that EmberModule has the expected pytree methods.

    EmberModule should have __pytree_flatten__ and __pytree_unflatten__ methods
    for integration with the tree transformation system.
    """
    assert hasattr(
        EmberModule, "__pytree_flatten__"
    ), "EmberModule should have __pytree_flatten__ method"
    assert callable(
        EmberModule.__pytree_flatten__
    ), "__pytree_flatten__ should be callable"

    assert hasattr(
        EmberModule, "__pytree_unflatten__"
    ), "EmberModule should have __pytree_unflatten__ method"
    assert callable(
        EmberModule.__pytree_unflatten__
    ), "__pytree_unflatten__ should be callable"


def test_empty_embermodule_flattening() -> None:
    """Tests flattening of an empty EmberModule instance.

    An empty EmberModule instance should be flattenable without errors and
    produce an appropriate representation for use with tree transformations.
    """

    class EmptyModule(EmberModule):
        pass

    # Create an instance
    instance = EmptyModule()

    # Try to flatten it
    try:
        leaves, aux = tree_flatten(tree=instance)

        # If successful, verify some basics
        assert isinstance(leaves, list), "Flattened leaves should be a list"
        assert isinstance(aux, tuple), "Auxiliary data should be a tuple"
        assert len(aux) == 2, "Auxiliary data should be a tuple of length 2"
        assert (
            aux[0] is EmptyModule
        ), "First element of auxiliary data should be the module class"
    except Exception as e:
        pytest.fail(f"Failed to flatten empty EmberModule: {e}")


def test_tree_flattening() -> None:
    """Tests the tree_flatten utility function with various types.

    tree_flatten should work with lists, dictionaries, and EmberModule instances.
    """
    # Test with list
    my_list = [1, 2, 3]
    leaves, aux = tree_flatten(tree=my_list)
    assert leaves == [1, 2, 3], "List leaves should be the list elements"
    assert aux[0] is list, "List auxiliary data should contain the list type"

    # Test with dict
    my_dict = {"a": 1, "b": 2}
    leaves, aux = tree_flatten(tree=my_dict)
    assert set(leaves) == {1, 2}, "Dict leaves should be the dict values"
    assert aux[0] is dict, "Dict auxiliary data should contain the dict type"

    # Test with EmberModule
    class SimpleModule(EmberModule):
        pass

    my_module = SimpleModule()
    leaves, aux = tree_flatten(tree=my_module)
    assert (
        aux[0] is SimpleModule
    ), "Module auxiliary data should contain the module class"


def test_tree_unflatten() -> None:
    """Tests the tree_unflatten utility function with various types.

    tree_unflatten should be able to reconstruct lists and dictionaries from flattened data.
    """
    # Test with list
    my_list = [1, 2, 3]
    leaves, aux = tree_flatten(tree=my_list)
    reconstructed = tree_unflatten(aux=aux, children=leaves)
    assert reconstructed == my_list, "Reconstructed list should equal original"

    # Test with dict
    my_dict = {"a": 1, "b": 2}
    leaves, aux = tree_flatten(tree=my_dict)
    reconstructed = tree_unflatten(aux=aux, children=leaves)
    assert reconstructed == my_dict, "Reconstructed dict should equal original"


# -----------------------------------------------------------------------------
# Integration with Other Test Modules
# -----------------------------------------------------------------------------


def test_imported_modules() -> None:
    """Tests that the necessary modules can be imported for use with EmberModule.

    Tests should be able to import the modules needed for working with EmberModule.
    """
    # Check that dataclasses is imported
    assert "dataclasses" in sys.modules, "dataclasses module should be imported"

    # Check that tree_util functions are imported
    assert callable(tree_flatten), "tree_flatten should be imported and callable"
    assert callable(tree_unflatten), "tree_unflatten should be imported and callable"


def test_thread_local_initialization() -> None:
    """Tests that _thread_local is correctly initialized.

    _thread_local should be an instance of threading.local.
    """
    assert isinstance(
        _thread_local, threading.local
    ), "_thread_local should be an instance of threading.local"
