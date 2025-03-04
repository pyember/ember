"""
EmberModel - A unified type system for Ember that standardizes input/output models.
"""

from __future__ import annotations

from typing import (
    Any,
    ClassVar,
    Dict,
    Type,
    TypeVar,
    Optional,
    Union,
    List,
    get_type_hints,
    cast,
)
from pydantic import BaseModel, create_model

# Import locally to avoid circular imports
from .protocols import TypeInfo, EmberTyped, EmberSerializable

T = TypeVar("T", bound="EmberModel")


class EmberModel(BaseModel):
    """
    A unified model for Ember input/output types that combines BaseModel validation
    with flexible serialization to dict, JSON, and potentially other formats.

    This class supports both attribute access (model.attr) and dictionary access (model["attr"])
    patterns for maximum flexibility and backward compatibility.

    It implements EmberTyped and EmberSerializable protocols to provide consistent
    type information and serialization capabilities.
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
        # Use model_validate to properly handle the conversion from dict to model
        # This properly handles type constraints in a way that the direct constructor may not
        # Workaround for pydantic's model_validate typing issues by creating first with constructor
        # then validating with model_validate
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
        Enables backward compatibility with code expecting different return types.
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

    # Dynamic model creation
    @classmethod
    def create_type(
        cls, name: str, fields: Dict[str, Type[object]], output_format: str = "model"
    ) -> Type["EmberModel"]:
        """
        Dynamically create a new EmberModel subclass with the specified fields.

        Args:
            name: Name of the model class
            fields: Dictionary mapping field names to types
            output_format: Default output format ("model", "dict", or "json")

        Returns:
            A new EmberModel subclass
        """
        # Create field definitions with proper ellipsis for required fields
        field_definitions = {}
        for k, v in fields.items():
            field_definitions[k] = (v, ...)  # All fields are required by default
            
        # Use dict-based approach to work around typing limitations
        # This approach creates the model directly with appropriate base class
        # without using create_model which has typing constraints
        model_attrs = {
            "__annotations__": {k: v for k, v in fields.items()},
            "__module__": __name__,
            "__doc__": f"Dynamically generated EmberModel: {name}"
        }
        
        # Create the model class directly as a subclass
        model_class = type(name, (cls,), model_attrs)

        # Set the output format
        setattr(model_class, '__output_format__', output_format)
        
        # Explicitly cast to the correct return type
        return cast(Type["EmberModel"], model_class)
