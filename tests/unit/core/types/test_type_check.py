"""
Tests for the type checking utilities.
"""

import pytest
from typing import Dict, List, Optional, Union, Any, TypeVar, Generic, Tuple

from ember.core.types.type_check import (
    validate_type,
    validate_instance_attrs,
    type_check,
)
from ember.core.types.ember_model import EmberModel
from ember.core.types.protocols import EmberTyped, TypeInfo


def test_validate_type_simple():
    """Test validation of simple types."""
    assert validate_type(42, int)
    assert not validate_type("42", int)
    assert validate_type("hello", str)
    assert validate_type(3.14, float)
    assert validate_type(True, bool)


def test_validate_type_optional():
    """Test validation of Optional types."""
    assert validate_type(None, Optional[int])
    assert validate_type(42, Optional[int])
    assert not validate_type("42", Optional[int])


def test_validate_type_union():
    """Test validation of Union types."""
    assert validate_type(42, Union[int, str])
    assert validate_type("42", Union[int, str])
    assert not validate_type(3.14, Union[int, str])


def test_validate_type_containers():
    """Test validation of container types like List and Dict."""
    assert validate_type([1, 2, 3], List[int])
    assert not validate_type([1, "2", 3], List[int])
    assert validate_type({"a": 1, "b": 2}, Dict[str, int])
    assert not validate_type({"a": 1, "b": "2"}, Dict[str, int])
    assert validate_type((1, "a"), Tuple[int, str])
    assert not validate_type((1, 2), Tuple[int, str])


class SimpleClass:
    """A simple class for testing attribute validation."""

    def __init__(self, a: int, b: str):
        self.a = a
        self.b = b


def test_validate_instance_attrs():
    """Test validation of object attributes."""
    obj = SimpleClass(a=42, b="hello")
    assert validate_instance_attrs(obj, SimpleClass) == {}

    # Test with invalid attributes
    obj.a = "42"  # type: ignore
    errors = validate_instance_attrs(obj, SimpleClass)
    assert "a" in errors


class ModelWithTypes(EmberModel):
    """A model with type annotations for testing."""

    a: int
    b: str
    c: Optional[List[Dict[str, Any]]] = None


def test_type_check():
    """Test the combined type_check function."""
    model = ModelWithTypes(a=42, b="hello")
    assert type_check(model, ModelWithTypes)

    # Test with invalid model
    model.a = "42"  # type: ignore
    assert not type_check(model, ModelWithTypes)

    # Test with simple types
    assert type_check(42, int)
    assert not type_check("42", int)


def test_protocol_checking():
    """Test checking against protocols."""

    class MyClass:
        def get_type_info(self) -> TypeInfo:
            return TypeInfo(origin_type=type(self))

    obj = MyClass()
    assert validate_type(obj, EmberTyped)


T = TypeVar("T")


class GenericContainer(Generic[T]):
    """A generic container for testing."""

    def __init__(self, value: T):
        self.value = value


def test_generic_types():
    """Test validation with generic types."""
    int_container = GenericContainer[int](42)
    str_container = GenericContainer[str]("hello")

    # This is a limitation of runtime type checking - we can't validate
    # the type parameter T at runtime without additional machinery
    assert validate_type(int_container, GenericContainer)
    assert validate_type(str_container, GenericContainer)
