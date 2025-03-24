"""
EmberModel - Base class for validated data models in the Ember framework.

Provides a consistent foundation for all data models with validation,
serialization, and type inspection capabilities.
"""

from __future__ import annotations

import json
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)

from pydantic import BaseModel, ConfigDict, Field

# Import locally to avoid circular imports
from .protocols import Serializable, TypedProtocol, TypeInfo

T = TypeVar("T", bound="EmberModel")


class EmberModel(BaseModel):
    """
    Base class for all data models in the Ember framework.

    Combines Pydantic's validation with consistent serialization capabilities,
    serving as the foundation for data structures throughout the framework.

    Features:
    - Strong validation through Pydantic
    - Consistent serialization to/from different formats
    - Type introspection for generic programming
    - Dictionary-like access for compatibility
    """

    # Use the new ConfigDict style for Pydantic v2 compatibility
    model_config = ConfigDict(extra="forbid")

    # TypedProtocol implementation
    def get_type_info(self) -> TypeInfo:
        """
        Return metadata about this model's type structure.

        Analyzes type annotations to provide runtime type information.

        Returns:
            TypeInfo with details about this model's type structure
        """
        type_hints = get_type_hints(self.__class__)
        return TypeInfo(
            origin_type=self.__class__,
            type_args=tuple(type_hints.values()) if type_hints else None,
            is_container=False,
            is_optional=False,
        )

    # Serializable protocol implementation
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this model to a dictionary representation.

        Returns:
            Dict representation of this model
        """
        return self.model_dump()

    def to_json(self) -> str:
        """
        Convert this model to a JSON string representation.

        Returns:
            JSON string representation of this model
        """
        return self.model_dump_json()

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Create a model instance from a dictionary.

        Args:
            data: Dictionary containing field values

        Returns:
            Validated instance of this model class

        Raises:
            ValidationError: If data doesn't match the model's schema
        """
        return cls.model_validate(data)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """
        Create a model instance from a JSON string.

        Args:
            json_str: JSON string containing field values

        Returns:
            Validated instance of this model class

        Raises:
            ValidationError: If data doesn't match the model's schema
            JSONDecodeError: If the JSON string is invalid
        """
        return cls.from_dict(json.loads(json_str))

    # Dictionary-like access for backward compatibility
    def __getitem__(self, key: str) -> Any:
        """
        Enable dictionary-like access to model attributes.

        Args:
            key: Attribute name to access

        Returns:
            Value of the requested attribute

        Raises:
            KeyError: If the attribute doesn't exist
        """
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __call__(self) -> Any:
        """
        Return model in the format specified by set_output_format.

        Returns:
            The model in the specified format (default is self).
        """
        format_type = getattr(self, "_output_format", "model")

        if format_type == "dict":
            return self.to_dict()
        elif format_type == "json":
            return self.to_json()
        else:  # Default to model
            return self

    def set_output_format(self, format_type: str) -> None:
        """
        Set the output format for when the model is called.

        Args:
            format_type: The output format to use ("dict", "json", or "model").
        """
        self._output_format = format_type

    # Dynamic model creation
    @classmethod
    def create_type(cls, name: str, fields: Dict[str, Type[Any]]) -> Type["EmberModel"]:
        """
        Dynamically create a new EmberModel subclass with specified fields.

        Creates a model class at runtime for dynamic schema support.

        Args:
            name: Name for the new model class
            fields: Dictionary mapping field names to types

        Returns:
            A new EmberModel subclass with the specified fields
        """
        # Create field definitions with proper ellipsis for required fields
        field_definitions = {}
        for k, v in fields.items():
            field_definitions[k] = (v, ...)  # All fields are required by default

        # Use dict-based approach to work around typing limitations
        model_attrs = {
            "__annotations__": {k: v for k, v in fields.items()},
            "__module__": __name__,
            "__doc__": f"Dynamically generated EmberModel: {name}",
        }

        # Create the model class directly as a subclass
        model_class = type(name, (cls,), model_attrs)

        # Explicitly cast to the correct return type
        return cast(Type["EmberModel"], model_class)

    # Backward compatibility methods
    def as_dict(self) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.

        Returns:
            Dict representation of this model
        """
        return self.to_dict()

    def as_json(self) -> str:
        """
        Legacy method for backward compatibility.

        Returns:
            JSON string representation of this model
        """
        return self.to_json()
