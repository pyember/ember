"""
EmberModel - A unified type system for Ember that standardizes input/output models.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Type, TypeVar, Optional, Union, List, get_type_hints
from pydantic import BaseModel, create_model

# Import locally to avoid circular imports
from .protocols import TypeInfo, EmberTyped, EmberSerializable

T = TypeVar('T', bound='EmberModel')


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
            raise ValueError(f"Unsupported format: {format}. Use 'model', 'dict', or 'json'")
        cls.__output_format__ = format
        
    def set_output_format(self, format: str) -> None:
        """Set the output format for this specific instance."""
        if format not in ["model", "dict", "json"]:
            raise ValueError(f"Unsupported format: {format}. Use 'model', 'dict', or 'json'")
        self._instance_output_format = format
        
    @property
    def output_format(self) -> str:
        """Get the effective output format for this instance."""
        return self._instance_output_format or self.__output_format__
    
    # EmberSerializable protocol implementation
    def as_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        return self.model_dump()
    
    def as_json(self) -> str:
        """Convert to a JSON string."""
        return self.model_dump_json()
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create an instance from a dictionary."""
        return cls(**data)
    
    # EmberTyped protocol implementation
    def get_type_info(self) -> TypeInfo:
        """Return type metadata for this object."""
        type_hints = get_type_hints(self.__class__)
        return TypeInfo(
            origin_type=self.__class__,
            type_args=tuple(type_hints.values()) if type_hints else None,
            is_container=False,
            is_optional=False
        )
    
    # Compatibility operators
    def __call__(self) -> Union[Dict[str, Any], str, 'EmberModel']:
        """
        Return the model in the configured format when called as a function.
        Enables backward compatibility with code expecting different return types.
        """
        match self.output_format:
            case "dict":
                return self.as_dict()
            case "json":
                return self.as_json()
            case _:
                return self
            
    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-like access (model["attr"]) alongside attribute access (model.attr)."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)
    
    # Dynamic model creation
    @classmethod
    def create_type(
        cls, 
        name: str, 
        fields: Dict[str, Type[Any]], 
        output_format: str = "model"
    ) -> Type[EmberModel]:
        """
        Dynamically create a new EmberModel subclass with the specified fields.
        
        Args:
            name: Name of the model class
            fields: Dictionary mapping field names to types
            output_format: Default output format ("model", "dict", or "json")
            
        Returns:
            A new EmberModel subclass
        """
        model_class = create_model(
            name,
            __base__=EmberModel,
            **{k: (v, ...) for k, v in fields.items()}
        )
        
        model_class.__output_format__ = output_format
        return model_class