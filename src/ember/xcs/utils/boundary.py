"""
Boundary Layer between XCS Execution Engine and Ember Operator System.

This module creates an explicit boundary between the dictionary-based XCS execution 
engine and the EmberModel-based operator system, ensuring type consistency.
"""

import inspect
import logging
from typing import Any, Dict, Optional, Type, TypeVar, Union

from ember.core.types.ember_model import EmberModel

logger = logging.getLogger(__name__)

T = TypeVar('T')


def to_ember_model(data: Dict[str, Any], model_type: Type[T]) -> T:
    """
    Convert XCS dictionary data to Ember model at system boundary.
    
    A single, consistent entry point for all XCS → Ember conversions.
    
    Args:
        data: Dictionary data from XCS execution engine
        model_type: Target Ember model type
        
    Returns:
        Instance of the specified model type
        
    Raises:
        TypeError: If conversion fails
    """
    if model_type is None:
        raise TypeError("Model type must be specified for boundary crossing")
        
    # Handle the case where data is already the right type
    if isinstance(data, model_type):
        return data
        
    # Regular conversion path for dictionaries
    if isinstance(data, dict):
        # Use model_validate from the model_type (based on pydantic)
        try:
            if hasattr(model_type, "model_validate"):
                result = model_type.model_validate(data)
                logger.debug(f"Boundary crossing: converted {type(data).__name__} → {model_type.__name__}")
                return result
        except Exception as e:
            logger.debug(f"Boundary crossing failed: {e}")
    
    # Handle nested structures by recursively traversing them
    if isinstance(data, dict) and hasattr(model_type, "__annotations__"):
        try:
            # First attempt the top-level conversion
            if hasattr(model_type, "model_validate"):
                result = model_type.model_validate(data)
                
                # Then handle nested fields by inspecting annotations
                annotations = model_type.__annotations__
                for field_name, field_type in annotations.items():
                    if not hasattr(result, field_name):
                        continue
                        
                    field_value = getattr(result, field_name)
                    
                    # Handle lists of models
                    if hasattr(field_type, "__origin__") and field_type.__origin__ == list:
                        if hasattr(field_type, "__args__") and len(field_type.__args__) > 0:
                            item_type = field_type.__args__[0]
                            
                            # Only attempt conversion on list items if they look like they need it
                            if field_value and isinstance(field_value, list):
                                converted_list = []
                                for item in field_value:
                                    if isinstance(item, dict) and hasattr(item_type, "model_validate"):
                                        try:
                                            converted_list.append(item_type.model_validate(item))
                                        except Exception:
                                            # Fallback to the original item if conversion fails
                                            converted_list.append(item)
                                    else:
                                        converted_list.append(item)
                                setattr(result, field_name, converted_list)
                
                return result
        except Exception as e:
            logger.debug(f"Nested model conversion failed: {e}")
    
    # If all conversions fail, raise a clear error
    raise TypeError(
        f"Boundary crossing failed: Could not convert {type(data).__name__} to {model_type.__name__}. "
        f"Keys available: {sorted(data.keys() if isinstance(data, dict) else [])}"
    )


def debug_type_conversion(name: str, value: Any, stage: str) -> None:
    """Log detailed type information for debugging boundary issues.
    
    Args:
        name: Name of the object being logged (e.g., "inputs", "outputs")
        value: The value to log type information for
        stage: Processing stage (e.g., "before_conversion", "after_conversion")
    """
    logger = logging.getLogger("ember.xcs.boundary")
    
    if not logger.isEnabledFor(logging.DEBUG):
        return
        
    type_name = type(value).__name__
    module_name = type(value).__module__
    
    # Log basic type information
    logger.debug(f"[BOUNDARY] {name} ({stage}): type={type_name}, module={module_name}")
    
    # For dictionaries, log keys
    if isinstance(value, dict):
        logger.debug(f"[BOUNDARY] {name} keys: {sorted(value.keys())}")
        
    # For EmberModel instances, log attributes and methods
    elif hasattr(value, "__dict__"):
        attrs = {
            name: type(getattr(value, name)).__name__
            for name in dir(value)
            if not name.startswith("_") and not inspect.ismethod(getattr(value, name))
        }
        logger.debug(f"[BOUNDARY] {name} attributes: {attrs}")


def to_dict(model: Union[EmberModel, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert Ember model to XCS dictionary at system boundary.
    
    A single, consistent entry point for all Ember → XCS conversions.
    
    Args:
        model: Ember model or dictionary
        
    Returns:
        Dictionary representation for XCS
    """
    if isinstance(model, dict):
        return model
        
    if isinstance(model, EmberModel):
        return model.to_dict()
        
    # For other types with to_dict method
    if hasattr(model, "to_dict") and callable(getattr(model, "to_dict")):
        return model.to_dict()
        
    # Try __dict__ as fallback
    if hasattr(model, "__dict__"):
        return {k: v for k, v in model.__dict__.items() if not k.startswith("_")}
        
    # Last resort - try to convert simple objects directly
    if hasattr(model, "__slots__"):
        return {slot: getattr(model, slot) for slot in model.__slots__ if hasattr(model, slot)}
        
    # If all else fails, raise a clear error
    raise TypeError(f"Cannot convert {type(model).__name__} to dictionary at system boundary")