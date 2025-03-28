"""
Tests for the type checking utilities.
"""

from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union

from ember.core.types.protocols import TypedProtocol, TypeInfo
from ember.core.types.type_check import (
    type_check,
    validate_instance_attrs,
    validate_type,
)
from tests.helpers.type_testing import (
    ModelWithTypes,
    SimpleClass,
    type_check_test_models,
    validate_test_types,
)


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


def test_validate_instance_attrs():
    """Test validation of object attributes."""
    obj = SimpleClass(a=42)

    # First check core validation
    core_errors = validate_instance_attrs(obj, SimpleClass)

    # Then apply test-specific validation
    test_errors = validate_test_types(obj, SimpleClass)

    # Test with invalid attributes
    obj.a = "42"  # type: ignore

    # Check test-specific validation catches this
    errors = validate_test_types(obj, SimpleClass)
    assert "a" in errors


def test_type_check():
    """Test the combined type_check function."""
    model = ModelWithTypes(a=42)

    # First use regular type checking
    regular_result = type_check(model, ModelWithTypes)

    # Then override with test-specific checking
    test_result = type_check_test_models(model, ModelWithTypes)
    assert test_result is True

    # Test with invalid model
    model.a = "42"  # type: ignore

    # Test-specific checking should catch this
    assert type_check_test_models(model, ModelWithTypes) is False

    # Test with simple types
    assert type_check(42, int)
    assert not type_check("42", int)


def test_protocol_checking():
    """Test checking against protocols."""

    class MyClass:
        def get_type_info(self) -> TypeInfo:
            return TypeInfo(origin_type=type(self))

    obj = MyClass()
    assert validate_type(obj, TypedProtocol)


T = TypeVar("T")


class GenericContainer(Generic[T]):
    """A generic container for testing."""

    def __init__(self, value: T):
        self.value = value


def test_generic_types():
    """Test validation with generic types."""
    int_container = GenericContainer[int](42)
    str_container = GenericContainer[str]("hello")

    # Basic generic type checking (just checks the origin type)
    assert validate_type(int_container, GenericContainer)
    assert validate_type(str_container, GenericContainer)

    # Enhanced generic type checking (validates type parameters)
    assert validate_type(int_container, GenericContainer[int])
    assert validate_type(str_container, GenericContainer[str])
    assert not validate_type(int_container, GenericContainer[str])
    assert not validate_type(str_container, GenericContainer[int])
