"""Types API for Ember.

This module provides types used throughout the Ember API surface area.
It centralizes type definitions to ensure consistency and compatibility
across the codebase.

Examples:
    from ember.api.types import EmberModel, Field

    class MyInputModel(EmberModel):
        text: str = Field(description="Input text")
"""

from typing import Any, ClassVar, Dict, Generic, List, Optional, Type, TypeVar, Union

from ember.core.types import EmberSerializable, EmberTyped, InputT, OutputT, TypeInfo

# Re-export core types
from ember.core.types.ember_model import EmberModel, Field


# Utility function for extracting values from various response types
def extract_value(response: Any, key: str, default: Any = None) -> Any:
    """Extract a value from a response object, with fallbacks for different formats.

    This utility handles different response formats that might be returned from
    various model providers and extracts the specified key.

    Args:
        response: The response object to extract from (dict, object, etc.)
        key: The key to extract
        default: Default value if key is not found

    Returns:
        The extracted value, or the default if not found
    """
    # Try direct dictionary access
    if isinstance(response, dict) and key in response:
        return response[key]

    # Try attribute access
    if hasattr(response, key):
        return getattr(response, key)

    # Try data dictionary if it exists
    if (
        hasattr(response, "data")
        and isinstance(response.data, dict)
        and key in response.data
    ):
        return response.data[key]

    # Try to see if it's a nested structure
    if isinstance(response, dict):
        for k, v in response.items():
            if isinstance(v, dict) and key in v:
                return v[key]

    # Return default if all else fails
    return default


__all__ = [
    # Base types
    "EmberModel",  # Base model for input/output types
    "Field",  # Field for model definitions
    # Type variables
    "InputT",  # Type variable for operator inputs
    "OutputT",  # Type variable for operator outputs
    # Protocols
    "EmberTyped",
    "EmberSerializable",
    "TypeInfo",
    # Utility functions
    "extract_value",  # Extract values from response objects
    # Re-exported typing primitives
    "Any",
    "Dict",
    "List",
    "Optional",
    "TypeVar",
    "Union",
    "Generic",
    "ClassVar",
    "Type",
]
