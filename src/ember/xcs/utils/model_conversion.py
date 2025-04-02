"""Model conversion utilities for boundary consistency.

Provides utilities for maintaining type consistency between system boundaries, 
particularly at the interfaces between the XCS execution engine (which works with 
dictionaries) and Ember operators (which work with EmberModel objects).
"""

import inspect
import logging
from typing import Any, Dict, Optional, Type, TypeVar, get_type_hints

T = TypeVar('T')
logger = logging.getLogger(__name__)


def ensure_model_type(value: Any, model_type: Optional[Type[T]] = None) -> Any:
    """Ensures a value has the expected model type by converting if necessary.
    
    This is a central utility for enforcing type contracts across system boundaries.
    
    Args:
        value: Value to check and potentially convert
        model_type: Expected model type
        
    Returns:
        Value converted to the expected type if possible, otherwise the original value
    """
    # Debug log for better tracing
    logger.debug(f"ensure_model_type: value={type(value)}, model_type={model_type.__name__ if model_type else None}")
    
    # If value is already the right type, just return it
    if model_type and isinstance(value, model_type):
        logger.debug("Value already has correct type")
        return value
    
    # Only try conversion with both dict value and target model type
    if model_type is None:
        logger.debug("No model type provided")
        return value
    
    # Check if this is a dictionary or has dict-like attributes
    if isinstance(value, dict):
        # Debug the dict keys for better diagnosis
        logger.debug(f"Dict keys: {list(value.keys())}")
        
        # Check if the model type has the proper validation method
        try:
            if hasattr(model_type, "model_validate"):
                logger.debug(f"Converting dict to {model_type.__name__}")
                converted = model_type.model_validate(value)
                logger.debug(f"Conversion successful: {type(converted)}")
                return converted
        except Exception as e:
            # Log but don't crash on conversion errors
            logger.debug(f"Model conversion failed for {model_type.__name__}: {e}")
    elif hasattr(value, "__dict__"):
        # Handle case where value is an object but not the correct type
        # This handles conversion between different EmberModel classes
        logger.debug(f"Value is object with __dict__: {type(value).__name__}")
        
        try:
            # Extract data from object
            data = {}
            # First try to get all attributes
            for attr, attr_value in inspect.getmembers(value):
                if not attr.startswith('_') and not inspect.ismethod(attr_value):
                    data[attr] = attr_value
                    
            # If above doesn't work, try __dict__
            if not data and hasattr(value, "__dict__"):
                data = value.__dict__
                
            logger.debug(f"Extracted data: keys={list(data.keys())}")
            
            # Convert to target model
            if hasattr(model_type, "model_validate"):
                logger.debug(f"Converting object {type(value).__name__} to {model_type.__name__}")
                converted = model_type.model_validate(data)
                logger.debug(f"Object conversion successful: {type(converted)}")
                return converted
        except Exception as e:
            logger.debug(f"Object conversion failed: {e}")
    else:
        logger.debug(f"Value is not dict or object: {type(value)}")
    
    # If all conversion methods fail, return original value
    return value


def deep_model_conversion(value: Any, target_type: Optional[Type] = None) -> Any:
    """Recursively converts nested structures to proper model types.
    
    Detects nested model types and recursively converts dictionaries to the
    appropriate model objects throughout the hierarchy.
    
    Args:
        value: Value to check and potentially convert
        target_type: Expected model type at this level
        
    Returns:
        Recursively converted model object(s) or original value
    """
    logger.debug(f"Deep model conversion: value={type(value)}, target={target_type.__name__ if target_type else None}")
    
    # Handle simple case first
    if target_type and isinstance(value, target_type):
        return value
        
    # Handle dictionary conversion
    if isinstance(value, dict) and target_type:
        # First convert the top-level object
        obj = ensure_model_type(value, target_type)
        
        if obj is value:  # If conversion failed, return original
            return value
            
        # Check if there are field annotations with nested model types
        if hasattr(target_type, "__annotations__"):
            # Get the type annotations
            annotations = target_type.__annotations__
            
            # For each annotated field, check if it needs conversion
            for field_name, field_type in annotations.items():
                # Skip if field doesn't exist on obj
                if not hasattr(obj, field_name):
                    continue
                    
                field_value = getattr(obj, field_name)
                
                # Handle list of models
                if hasattr(field_type, "__origin__") and field_type.__origin__ == list:
                    # Get the contained type for List[T]
                    if hasattr(field_type, "__args__") and len(field_type.__args__) > 0:
                        item_type = field_type.__args__[0]
                        
                        # Recursively convert each item in the list
                        if field_value and isinstance(field_value, list):
                            converted_list = [deep_model_conversion(item, item_type) for item in field_value]
                            setattr(obj, field_name, converted_list)
                
                # Direct conversion for single objects
                elif field_value is not None:
                    # Check if this might be a model type
                    if hasattr(field_type, "model_validate"):
                        setattr(obj, field_name, deep_model_conversion(field_value, field_type))
        
        return obj
    
    # Handle list conversion without known target_type
    if isinstance(value, list):
        # Try best-effort conversion of list items
        return [deep_model_conversion(item, None) for item in value]
        
    return value