"""
Stub implementation of EmberModel and related protocols for testing.

This module provides stub implementations of EmberModel and related protocols
to support tests that depend on these interfaces without requiring the full
implementation.
"""

from __future__ import annotations

from typing import (
    ClassVar,
    Dict,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_type_hints,
    runtime_checkable,
)

from pydantic import BaseModel


# Define the protocols first (simplified versions)
@runtime_checkable
class EmberSerializable(Protocol):
    """Protocol for objects that can be serialized to/from dict and JSON formats."""

    def as_dict(self) -> Dict[str, object]:
        """Convert to a dictionary representation."""
        ...

    def as_json(self) -> str:
        """Convert to a JSON string."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "EmberSerializable":
        """Create an instance from a dictionary."""
        ...


class TypeInfo:
    """Information about a type for Ember type system."""

    def __init__(
        self,
        origin_type: Type,
        type_args: Optional[Tuple[Type, ...]] = None,
        is_container: bool = False,
        is_optional: bool = False,
    ):
        self.origin_type = origin_type
        self.type_args = type_args
        self.is_container = is_container
        self.is_optional = is_optional


@runtime_checkable
class EmberTyped(Protocol):
    """Protocol for objects that provide type information."""

    def get_type_info(self) -> TypeInfo:
        """Return type metadata for this object."""
        ...


# Create the EmberModel stub
T = TypeVar("T", bound="EmberModel")


class EmberModel(BaseModel):
    """
    A unified model for Ember input/output types that combines BaseModel validation
    with flexible serialization to dict, JSON, and potentially other formats.

    This is a stub implementation for testing.
    """

    # Class variable to store output format preference
    __output_format__: ClassVar[str] = "model"  # Options: "model", "dict", "json"

    # Instance variable for per-instance output format override
    _instance_output_format: Optional[str] = None

    @classmethod
    def set_default_output_format(cls, format: str) -> None:
        """Set the default output format for all EmberModel instances."""
        if format not in ["model", "dict", "json"]:
            raise ValueError(
                f"Unsupported format: {format}. Use 'model', 'dict', or 'json'"
            )
        cls.__output_format__ = format

    def set_output_format(self, format: str) -> None:
        """Set the output format for this specific instance."""
        if format not in ["model", "dict", "json"]:
            raise ValueError(
                f"Unsupported format: {format}. Use 'model', 'dict', or 'json'"
            )
        self._instance_output_format = format

    @property
    def output_format(self) -> str:
        """Get the effective output format for this instance."""
        return self._instance_output_format or self.__output_format__

    # EmberSerializable protocol implementation
    def as_dict(self) -> Dict[str, object]:
        """Convert to a dictionary representation."""
        return self.model_dump()

    def as_json(self) -> str:
        """Convert to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, object]) -> T:
        """Create an instance from a dictionary."""
        result = cls.model_validate(data)
        return result

    # EmberTyped protocol implementation
    def get_type_info(self) -> TypeInfo:
        """Return type metadata for this object."""
        type_hints = get_type_hints(self.__class__)
        return TypeInfo(
            origin_type=self.__class__,
            type_args=tuple(type_hints.values()) if type_hints else None,
            is_container=False,
            is_optional=False,
        )

    # Compatibility operators
    def __call__(self) -> Union[Dict[str, object], str, "EmberModel"]:
        """
        Return the model in the configured format when called as a function.
        """
        format_type = self.output_format
        if format_type == "dict":
            return self.as_dict()
        elif format_type == "json":
            return self.as_json()
        else:
            return self

    def __getitem__(self, key: str) -> object:
        """Enable dictionary-like access (model["attr"]) alongside attribute access (model.attr)."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)
