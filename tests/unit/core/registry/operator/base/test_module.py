#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the EmberModule abstraction.

This test suite provides comprehensive tests for the EmberModule abstraction,
its associated field utilities, the temporary mutable initialization wrapper, tree
flattening/unflattening functionality, bound method behavior, and metaclass registration.

These tests adhere to the Google Python Style Guide and enforce strong type annotations.
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
# Import the module itself to allow patching attributes.
from src.ember.core.registry.operator.base import _module


###############################################################################
# NOTE: EmberModuleMeta decorates subclasses as frozen dataclasses.
# Due to dataclass inheritance subtleties, dummy test classes explicitly use
# @dataclasses.dataclass(frozen=True, init=True) to ensure that __init__ accepts
# keyword arguments.
###############################################################################


class TestFieldFunctions(unittest.TestCase):
    """Unit tests for the static_field and ember_field utility functions."""

    def test_static_field_metadata(self) -> None:
        """Verify that static_field produces a field with 'static' metadata set to True.

        Raises:
            AssertionError: If the field metadata or default value is incorrect.
        """
        field_obj = static_field(default=10)
        self.assertEqual(first=field_obj.metadata.get("static"),
                         second=True,
                         msg="Expected static metadata to be True")
        self.assertEqual(first=field_obj.default,
                         second=10,
                         msg="Expected default value to be 10")

    def test_ember_field_with_converter(self) -> None:
        """Verify that ember_field adds a converter and static metadata when specified.

        Raises:
            AssertionError: If converter metadata, static metadata, or default value does not match expectations.
        """
        converter_fn: Callable[[Any], int] = lambda x: int(x)
        field_obj = ember_field(converter=converter_fn, static=True, default="123")
        self.assertEqual(first=field_obj.metadata.get("converter"),
                         second=converter_fn,
                         msg="Expected converter metadata to match the provided function")
        self.assertEqual(first=field_obj.metadata.get("static"),
                         second=True,
                         msg="Expected static metadata to be True")
        self.assertEqual(first=field_obj.default,
                         second="123",
                         msg="Expected default value to be '123'")

    def test_ember_field_without_converter(self) -> None:
        """Verify that ember_field does not add converter or static metadata when not provided.

        Raises:
            AssertionError: If converter or static metadata is incorrectly set or default value does not match.
        """
        field_obj = ember_field(default=5)
        self.assertIsNone(field_obj.metadata.get("converter"),
                          msg="Expected converter metadata to be None")
        self.assertIsNone(field_obj.metadata.get("static"),
                          msg="Expected static metadata to be None")
        self.assertEqual(first=field_obj.default,
                         second=5,
                         msg="Expected default value to be 5")

    def test_ember_field_converter_application(self) -> None:
        """Verify that ember_field converter is applied during initialization."""
        @dataclasses.dataclass(frozen=True)
        class ConverterModule(EmberModule):
            x: int = ember_field(converter=lambda v: int(v))

        instance = ConverterModule(x="42")
        self.assertEqual(
            instance.x,
            42,
            "Converter should convert string '42' to int 42"
        )


class TestInitableWrapper(unittest.TestCase):
    """Unit tests for the _make_initable_wrapper functionality."""

    def test_mutable_during_init_and_then_freeze(self) -> None:
        """Ensure a frozen dataclass wrapped via _make_initable_wrapper remains mutable during initialization,
        then becomes immutable afterward.

        Raises:
            AssertionError: If mutability is not as expected.
            FrozenInstanceError: If mutation is allowed after freezing.
        """
        @dataclasses.dataclass(frozen=True)
        class FrozenDummy:
            a: int

        # Create a mutable wrapper for FrozenDummy.
        MutableDummy: Type[FrozenDummy] = _make_initable_wrapper(FrozenDummy)
        instance = MutableDummy(a=1)  # type: ignore

        # Mutation is allowed during the mutable initialization phase.
        instance.a = 2
        self.assertEqual(first=instance.a,
                         second=2,
                         msg="Expected value of 'a' to be updated to 2 during initialization phase")

        # Revert the instance class back to the immutable FrozenDummy.
        object.__setattr__(instance, "__class__", FrozenDummy)
        with self.assertRaises(FrozenInstanceError,
                               msg="Mutation after freezing should raise FrozenInstanceError"):
            instance.a = 3


class TestFlattenAndUnflatten(unittest.TestCase):
    """Unit tests for _flatten_ember_module and _unflatten_ember_module functions."""

    def setUp(self) -> None:
        """Set up a dummy EmberModule subclass for flatten/unflatten tests."""
        class DummyModule(EmberModule):
            a: int = ember_field()
            b: int = ember_field(static=True)

        # Ensure a correct __init__ signature by decorating with dataclass.
        DummyModule = dataclasses.dataclass(frozen=True, init=True)(DummyModule)
        self.DummyModule: Type[DummyModule] = DummyModule  # type: ignore
        self.instance: DummyModule = DummyModule(a=10, b=20)

    def test_flatten_ember_module(self) -> None:
        """Verify that _flatten_ember_module correctly separates dynamic and static fields.

        Raises:
            AssertionError: If field separation does not match expected dynamic and static values.
        """
        dynamic, static = _flatten_ember_module(self.instance)
        self.assertEqual(first=dynamic,
                         second=[10],
                         msg="Expected dynamic fields to contain [10]")
        self.assertEqual(first=static,
                         second={"b": 20},
                         msg="Expected static fields to contain {'b': 20}")

    def test_unflatten_ember_module(self) -> None:
        """Verify that _unflatten_ember_module can reconstruct an instance correctly.

        Raises:
            AssertionError: If the reconstructed instance does not match the original's flattened representation.
        """
        aux: Dict[str, Any] = {"b": 20}
        children: List[Any] = [10]
        instance_reconstructed = _unflatten_ember_module(
            cls=self.DummyModule, aux=aux, children=children
        )
        self.assertEqual(first=_flatten_ember_module(self.instance),
                         second=_flatten_ember_module(instance_reconstructed),
                         msg="Reconstructed instance does not match original after flatten/unflatten")

    def test_unflatten_error_on_mismatch(self) -> None:
        """Verify that _unflatten_ember_module raises ValueError when the children count mismatches.

        Raises:
            ValueError: When the provided children list length does not match the number of dynamic fields.
        """
        aux: Dict[str, Any] = {"b": 20}
        children: List[Any] = []  # Missing dynamic fields.
        with self.assertRaises(ValueError,
                               msg="Expected ValueError for mismatched dynamic field count"):
            _unflatten_ember_module(cls=self.DummyModule, aux=aux, children=children)

    def test_nested_module_flatten_unflatten(self) -> None:
        """Verify that nested EmberModule instances are flattened and unflattened correctly.

        Raises:
            AssertionError: If the reconstructed nested module does not match the original.
        """
        import dataclasses

        class InnerModule(EmberModule):
            x: int = ember_field()

        InnerModule = dataclasses.dataclass(frozen=True, init=True)(InnerModule)

        class OuterModule(EmberModule):
            inner: InnerModule = ember_field()
            y: int = ember_field(static=True)

        OuterModule = dataclasses.dataclass(frozen=True, init=True)(OuterModule)

        inner_instance = InnerModule(x=42)
        outer_instance = OuterModule(inner=inner_instance, y=100)

        dynamic, static = _flatten_ember_module(outer_instance)
        self.assertEqual(first=len(dynamic),
                         second=1,
                         msg="Expected exactly one dynamic field for nested modules")
        self.assertEqual(first=static,
                         second={"y": 100},
                         msg="Expected static fields to contain {'y': 100}")

        reconstructed = _unflatten_ember_module(cls=OuterModule, aux=static, children=dynamic)
        self.assertEqual(first=outer_instance,
                         second=reconstructed,
                         msg="Reconstructed nested module instance does not match the original")


class TestEmberModule(unittest.TestCase):
    """Unit tests for the EmberModule base class including initialization immutability and tree-based equality/hash semantics."""

    def setUp(self) -> None:
        """Initialize a dummy EmberModule subclass that uses _init_field in __post_init__."""
        class ComputedModule(EmberModule):
            a: int = ember_field()
            b: int = ember_field(init=False)

            def __post_init__(self) -> None:
                # Initialize field 'b' as twice the value of 'a'.
                self._init_field(field_name="b", value=self.a * 2)

        # Enforce a proper __init__ signature using the dataclass decorator.
        ComputedModule = dataclasses.dataclass(frozen=True, init=True)(ComputedModule)
        self.ComputedModule: Type[ComputedModule] = ComputedModule  # type: ignore

    def test_mutable_initialization_phase(self) -> None:
        """Ensure EmberModule permits mutable initialization before freezing.

        Raises:
            AssertionError: If the computed field or immutability post-initialization does not match expectations.
        """
        instance = self.ComputedModule(a=5)
        self.assertEqual(first=instance.b,
                         second=10,
                         msg="Field 'b' should be twice field 'a'")
        with self.assertRaises(FrozenInstanceError,
                               msg="Mutation should not be allowed post initialization"):
            instance.a = 20

    def test_hash_and_eq(self) -> None:
        """Verify that __hash__ and __eq__ use the tree-flattened representation.

        Patches tree_flatten in the module namespace to use _flatten_ember_module.

        Raises:
            AssertionError: If equality or hash semantics are inconsistent with expectations.
        """
        with patch.object(_module, "tree_flatten", new=_flatten_ember_module):
            instance1 = self.ComputedModule(a=7)
            instance2 = self.ComputedModule(a=7)
            instance3 = self.ComputedModule(a=8)
            self.assertEqual(first=instance1,
                             second=instance2,
                             msg="Instances with the same state should be equal")
            self.assertEqual(first=hash(instance1),
                             second=hash(instance2),
                             msg="Hashes should be identical for instances with the same state")
            self.assertNotEqual(first=instance1,
                                second=instance3,
                                msg="Instances with different state should not be equal")

    def test_repr(self) -> None:
        """Verify that __repr__ returns a string containing the class name and field dictionary.

        Raises:
            AssertionError: If the __repr__ output does not include the expected class name.
        """
        instance = self.ComputedModule(a=3)
        self.assertIn(member=self.ComputedModule.__name__ + "(",
                      container=instance.__repr__(),
                      msg="The repr output should include the class name and fields.")


class TestBoundMethod(unittest.TestCase):
    """Unit tests for the BoundMethod class functionality."""

    def test_bound_method_call_and_wrapped(self) -> None:
        """Verify that BoundMethod correctly invokes the underlying function and that __wrapped__
        returns the original bound function.

        Raises:
            AssertionError: If BoundMethod does not behave as expected.
        """
        def dummy_func(self_obj: Any, increment: int) -> int:
            return self_obj.value + increment

        class DummyModule(EmberModule):
            value: int = ember_field()

        DummyModule = dataclasses.dataclass(frozen=True, init=True)(DummyModule)
        dummy_instance = DummyModule(value=10)

        # Create a mutable wrapper for BoundMethod.
        MutableBoundMethod: Type[BoundMethod] = _make_initable_wrapper(BoundMethod)
        bound_method_instance = MutableBoundMethod()  # type: ignore
        object.__setattr__(bound_method_instance, "__func__", dummy_func)
        object.__setattr__(bound_method_instance, "__self__", dummy_instance)
        # Revert the instance back to the immutable BoundMethod class.
        object.__setattr__(bound_method_instance, "__class__", BoundMethod)

        result: int = bound_method_instance(5)
        self.assertEqual(first=result,
                         second=15,
                         msg="Expected bound method to return cumulative value (10+5=15)")
        wrapped = bound_method_instance.__wrapped__
        self.assertEqual(first=wrapped(7),
                         second=17,
                         msg="Expected wrapped function to yield correct result (10+7=17)")


class TestEmberModuleMetaRegistration(unittest.TestCase):
    """Unit tests to ensure that EmberModuleMeta registers new classes with the tree system."""

    def test_register_tree_called(self) -> None:
        """Verify that creating a new EmberModule subclass triggers a call to register_tree.

        Raises:
            AssertionError: If register_tree is not called or is invoked with incorrect arguments.
        """
        with patch.object(_module, "register_tree") as mock_register_tree:
            class MetaTestModule(EmberModule):
                x: int = ember_field()

            self.assertTrue(mock_register_tree.called,
                            msg="Expected register_tree to be called upon new subclass creation")
            args, kwargs = mock_register_tree.call_args
            self.assertIn("cls", kwargs,
                          msg="register_tree call should include 'cls' keyword argument")
            self.assertIn("flatten_func", kwargs,
                          msg="register_tree call should include 'flatten_func' keyword argument")
            self.assertIn("unflatten_func", kwargs,
                          msg="register_tree call should include 'unflatten_func' keyword argument")
            self.assertEqual(kwargs["cls"],
                             MetaTestModule,
                             msg="register_tree 'cls' argument should match the new subclass")


if __name__ == "__main__":
    unittest.main()