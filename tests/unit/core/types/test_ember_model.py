"""
Tests for EmberModel and its protocol implementations.
"""

import pytest
from typing import Dict, Any, List, Optional

from ember.core.types.ember_model import EmberModel
from ember.core.types.protocols import EmberTyped, EmberSerializable


class SampleModel(EmberModel):
    """Sample model for unit testing."""

    name: str
    value: int
    tags: Optional[List[str]] = None


def test_ember_model_creation():
    """Test creating and using EmberModel instances."""
    model = SampleModel(name="test", value=42)
    assert model.name == "test"
    assert model.value == 42
    assert model.tags is None


def test_ember_model_serialization():
    """Test serialization methods of EmberModel."""
    model = SampleModel(name="test", value=42, tags=["a", "b"])

    # Test as_dict() method
    model_dict = model.as_dict()
    assert isinstance(model_dict, dict)
    assert model_dict["name"] == "test"
    assert model_dict["value"] == 42
    assert model_dict["tags"] == ["a", "b"]

    # Test as_json() method
    model_json = model.as_json()
    assert isinstance(model_json, str)
    assert '"name":"test"' in model_json.replace(" ", "")
    assert '"value":42' in model_json.replace(" ", "")


def test_ember_model_protocol_implementation():
    """Test protocol implementation for EmberTyped and EmberSerializable."""
    model = SampleModel(name="test", value=42)

    # Test EmberTyped protocol
    assert isinstance(model, EmberTyped)
    type_info = model.get_type_info()
    assert type_info.origin_type == SampleModel

    # Test EmberSerializable protocol
    assert isinstance(model, EmberSerializable)
    model_dict = model.as_dict()
    assert isinstance(model_dict, dict)
    assert model_dict["name"] == "test"

    # Test from_dict static method
    new_model = SampleModel.from_dict({"name": "clone", "value": 100})
    assert new_model.name == "clone"
    assert new_model.value == 100


def test_ember_model_access_patterns():
    """Test that EmberModel supports both attribute and dictionary access patterns."""
    model = SampleModel(name="test", value=42, tags=["a", "b"])

    # Test attribute access
    assert model.name == "test"
    assert model.value == 42
    assert model.tags == ["a", "b"]

    # Test dictionary-like access
    assert model["name"] == "test"
    assert model["value"] == 42
    assert model["tags"] == ["a", "b"]

    # Test attribute error
    with pytest.raises(AttributeError):
        _ = model.nonexistent

    # Test with KeyError
    with pytest.raises(KeyError):
        _ = model["nonexistent"]


def test_ember_model_dynamic_creation():
    """Test dynamic creation of EmberModel subclasses."""
    fields = {"name": str, "count": int, "is_active": bool}

    DynamicModel = EmberModel.create_type("DynamicModel", fields)
    model = DynamicModel(name="dynamic", count=5, is_active=True)

    # Test attribute access
    assert model.name == "dynamic"
    assert model.count == 5
    assert model.is_active is True

    # Test dictionary-like access
    assert model["name"] == "dynamic"
    assert model["count"] == 5
    assert model["is_active"] is True

    # Verify class inheritance and protocol implementation
    assert isinstance(model, EmberModel)
    assert isinstance(model, EmberTyped)
    assert isinstance(model, EmberSerializable)
