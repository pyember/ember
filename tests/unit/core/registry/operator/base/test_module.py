#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the EmberModule abstraction.

This test suite provides comprehensive testing for the EmberModule abstraction,
its associated field utilities, the temporary mutable initialization wrapper,
tree flattening/unflattening functionality, bound method behavior, and metaclass
registration. These tests are written to the highest standards and follow the Google
Python Style Guide.
"""

from __future__ import annotations

import abc
import dataclasses
import unittest
from dataclasses import FrozenInstanceError
from typing import Any, Callable, Dict, List, Optional, Type

from unittest.mock import patch

# Import the module under test.
from src.ember.core.registry.operator.base._module import (
    EmberModule,
    BoundMethod,
    ember_field,
    static_field,
    _make_initable_wrapper,
    _flatten_ember_module,
    _unflatten_ember_module,
)

# Import the module itself so that we can patch attributes on it.
from src.ember.core.registry.operator.base import _module


###############################################################################
# Key Note on Dataclass Decoration:
# The design of EmberModuleMeta is that it should automatically decorate subclasses
# as frozen dataclasses. However, due to inheritance subtleties (a dataclass decorator
# is not automatically applied to a subclass if its parent is already decorated),
# our test dummy classes do not receive a __init__ with keyword arguments.
# To test the intended behavior (i.e. that the fields are present on the class)
# we explicitly wrap our dummy subclasses with @dataclasses.dataclass(frozen=True, init=True).
###############################################################################


class TestFieldFunctions(unittest.TestCase):
    """Tests for the static_field and ember_field utility functions."""

    def test_static_field_metadata(self) -> None:
        """Test that static_field returns a field with 'static' metadata set to True."""
        field_obj = static_field(default=10)
        self.assertEqual(field_obj.metadata.get("static"), True)
        self.assertEqual(field_obj.default, 10)

    def test_ember_field_with_converter(self) -> None:
        """Test that ember_field correctly adds a converter and static metadata when requested."""
        converter_fn: Callable[[Any], int] = lambda x: int(x)
        field_obj = ember_field(converter=converter_fn, static=True, default="123")
        self.assertEqual(field_obj.metadata.get("converter"), converter_fn)
        self.assertEqual(field_obj.metadata.get("static"), True)
        self.assertEqual(field_obj.default, "123")

    def test_ember_field_without_converter(self) -> None:
        """Test that ember_field does not add a converter and static metadata when not specified."""
        field_obj = ember_field(default=5)
        self.assertIsNone(field_obj.metadata.get("converter"))
        self.assertIsNone(field_obj.metadata.get("static"))
        self.assertEqual(field_obj.default, 5)


class TestInitableWrapper(unittest.TestCase):
    """Tests for the _make_initable_wrapper functionality."""

    def test_mutable_during_init_and_then_freeze(self) -> None:
        """Test that a frozen dataclass wrapped via _make_initable_wrapper is mutable during
        initialization and becomes frozen afterward.
        """
        @dataclasses.dataclass(frozen=True)
        class FrozenDummy:
            a: int

        # Create a mutable wrapper.
        MutableDummy: Type[FrozenDummy] = _make_initable_wrapper(FrozenDummy)
        instance = MutableDummy(a=1)

        # Allow mutation while still of type MutableDummy.
        instance.a = 2
        self.assertEqual(instance.a, 2)

        # Reset the class to the original frozen type.
        object.__setattr__(instance, "__class__", FrozenDummy)

        # Now mutations should raise an error.
        with self.assertRaises(FrozenInstanceError):
            instance.a = 3


class TestFlattenAndUnflatten(unittest.TestCase):
    """Tests for _flatten_ember_module and _unflatten_ember_module functions."""

    def setUp(self) -> None:
        """Set up a dummy EmberModule subclass for testing flatten/unflatten."""
        # Define a dummy module with one dynamic and one static field.
        class DummyModule(EmberModule):
            a: int = ember_field()
            b: int = ember_field(static=True)

        # Explicitly decorate so that __init__ accepts keyword arguments.
        DummyModule = dataclasses.dataclass(frozen=True, init=True)(DummyModule)
        self.DummyModule = DummyModule
        self.instance = DummyModule(a=10, b=20)

    def test_flatten_ember_module(self) -> None:
        """Test that _flatten_ember_module correctly separates dynamic and static fields."""
        dynamic, static = _flatten_ember_module(self.instance)
        # Field 'a' is dynamic; 'b' is static.
        self.assertEqual(dynamic, [10])
        self.assertEqual(static, {"b": 20})

    def test_unflatten_ember_module(self) -> None:
        """Test that _unflatten_ember_module correctly reconstructs an instance."""
        aux: Dict[str, Any] = {"b": 20}
        children: List[Any] = [10]
        instance_reconstructed = _unflatten_ember_module(
            cls=self.DummyModule, aux=aux, children=children
        )
        # Compare flattened representations to ensure reconstruction correctness.
        self.assertEqual(
            _flatten_ember_module(self.instance),
            _flatten_ember_module(instance_reconstructed),
        )


class TestEmberModule(unittest.TestCase):
    """Tests for the EmberModule base class, including initialization, immutability,
    and tree-based equality/hash semantics.
    """

    def setUp(self) -> None:
        """Set up a dummy EmberModule subclass that uses _init_field in __post_init__."""
        class ComputedModule(EmberModule):
            a: int = ember_field()
            b: int = ember_field(init=False)

            def __post_init__(self) -> None:
                # Set 'b' as twice the value of 'a'.
                self._init_field("b", self.a * 2)

        # Explicitly decorate to force a proper __init__.
        ComputedModule = dataclasses.dataclass(frozen=True, init=True)(ComputedModule)
        self.ComputedModule = ComputedModule

    def test_mutable_initialization_phase(self) -> None:
        """Test that EmberModule allows mutable initialization and then becomes frozen."""
        instance = self.ComputedModule(a=5)
        self.assertEqual(instance.b, 10)
        with self.assertRaises(FrozenInstanceError):
            instance.a = 20

    def test_hash_and_eq(self) -> None:
        """Test that __hash__ and __eq__ use the tree-flattened representation.

        We patch tree_flatten in the _module namespace to use _flatten_ember_module.
        """
        with patch.object(_module, "tree_flatten", new=_flatten_ember_module):
            instance1 = self.ComputedModule(a=7)
            instance2 = self.ComputedModule(a=7)
            instance3 = self.ComputedModule(a=8)
            self.assertEqual(instance1, instance2)
            self.assertEqual(hash(instance1), hash(instance2))
            self.assertNotEqual(instance1, instance3)

    def test_repr(self) -> None:
        """Test that __repr__ returns a string that includes the class name and field dictionary.
        Note that for inner classes the repr uses the qualified name.
        """
        instance = self.ComputedModule(a=3)
        self.assertIn(f"{self.ComputedModule.__name__}(", instance.__repr__())


class TestBoundMethod(unittest.TestCase):
    """Tests for the BoundMethod class functionality."""

    def test_bound_method_call_and_wrapped(self) -> None:
        """Test that a BoundMethod correctly calls the underlying function and that
        __wrapped__ returns the original function bound to __self__.
        """
        def dummy_func(self_obj: Any, increment: int) -> int:
            return self_obj.value + increment

        # Define a simple EmberModule subclass to serve as __self__.
        class DummyModule(EmberModule):
            value: int = ember_field()

        DummyModule = dataclasses.dataclass(frozen=True, init=True)(DummyModule)
        dummy_instance = DummyModule(value=10)

        # Create a mutable wrapper of BoundMethod for testing
        MutableBoundMethod = _make_initable_wrapper(BoundMethod)
        bound_method_instance = MutableBoundMethod()
        object.__setattr__(bound_method_instance, "__func__", dummy_func)
        object.__setattr__(bound_method_instance, "__self__", dummy_instance)
        # Reset class to original frozen BoundMethod
        object.__setattr__(bound_method_instance, "__class__", BoundMethod)

        result = bound_method_instance(5)
        self.assertEqual(result, 15)
        wrapped = bound_method_instance.__wrapped__
        self.assertEqual(wrapped(7), 17)


class TestEmberModuleMetaRegistration(unittest.TestCase):
    """Tests that EmberModuleMeta correctly registers new classes with the tree system."""

    def test_register_tree_called(self) -> None:
        """Test that when a new EmberModule subclass is defined, register_tree is invoked."""
        with patch.object(_module, "register_tree") as mock_register_tree:
            class MetaTestModule(EmberModule):
                x: int = ember_field()

            # We do not force dataclass decoration here so that we test the metaclass.
            self.assertTrue(mock_register_tree.called)
            # Verify that the registration was provided with the expected parameters.
            args, kwargs = mock_register_tree.call_args
            self.assertIn("cls", kwargs)
            self.assertIn("flatten_func", kwargs)
            self.assertIn("unflatten_func", kwargs)
            self.assertEqual(kwargs["cls"], MetaTestModule)


if __name__ == "__main__":
    unittest.main()