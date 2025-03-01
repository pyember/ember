"""
EmberModel - A unified type system for Ember that standardizes input/output models.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Type, TypeVar, Optional, Union, List
from pydantic import BaseModel, create_model

T = TypeVar('T', bound='EmberModel')


class EmberModel(BaseModel):
    """
    A unified model for Ember input/output types that combines BaseModel validation
    with flexible serialization to dict, JSON, and potentially other formats.
    
    This class can be configured to behave like a TypedDict when needed for
    backward compatibility with existing code.
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
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        return self.model_dump()
    
    def as_json(self) -> str:
        """Convert to a JSON string."""
        return self.model_dump_json()
    
    def __call__(self) -> Union[Dict[str, Any], str, 'EmberModel']:
        """
        Return the model in the default format when called as a function.
        This allows for backward compatibility with code that expects TypedDict.
        """
        if self.output_format == "dict":
            return self.as_dict()
        elif self.output_format == "json":
            return self.as_json()
        else:
            return self
            
    def __getitem__(self, key: str) -> Any:
        """
        Enable dictionary-like access for backward compatibility with TypedDict.
        This allows EmberModel to be used in places expecting a dict.
        """
        if key in self.model_fields_set:
            return getattr(self, key)
        raise KeyError(key)
        
    @classmethod
    def __get_validators__(cls):
        """
        Return validators that allow dict-to-model conversion during validation.
        This enables operators to return plain dictionaries that are compatible with EmberModel.
        """
        yield cls.validate
        
    @classmethod
    def validate(cls, value):
        """
        Validate the value, converting dictionaries to EmberModel instances.
        This allows operators to return simple dictionaries that are automatically
        converted to EmberModel instances by the type system.
        """
        if isinstance(value, dict):
            return cls(**value)
        elif isinstance(value, cls):
            return value
        raise TypeError(f"Cannot convert {type(value)} to {cls.__name__}")
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create an instance from a dictionary."""
        return cls(**data)
    
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
        # Create a new model class using Pydantic's create_model
        model_class = create_model(
            name,
            __base__=EmberModel,
            **{k: (v, ...) for k, v in fields.items()}
        )
        
        # Set the default output format
        model_class.__output_format__ = output_format
        
        return model_class