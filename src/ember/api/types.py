"""Shared type aliases, validators, and helpers exposed in ``ember.api``.

The module collects type-centric utilities into a single place so user code can
import validated models and helper functions without traversing internal
packages.

Examples:
    >>> from ember.api.types import EmberModel, field_validator
    >>>
    >>> class Prompt(EmberModel):
    ...     text: str
    ...
    ...     @field_validator("text")
    ...     def trim(cls, value: str) -> str:
    ...         return value.strip()
"""

from typing import Any, ClassVar, Dict, Generic, List, Optional, Type, TypeVar, Union

# Re-export core types
from ember._internal.types import EmberModel, Field

# Re-export validators for convenience
from ember.api.validators import field_validator, model_validator


# Utility function for extracting values from various response types
def extract_value(response: Any, key: str, default: Any = None) -> Any:
    """Extract ``key`` from heterogenous response payloads.

    Args:
        response: Object or mapping returned by a provider.
        key: Field name to retrieve.
        default: Value returned when the key cannot be found.

    Returns:
        Any: The resolved value or ``default`` if the key is missing.

    Examples:
        >>> response = {"text": "hi", "metadata": {"score": 0.5}}
        >>> extract_value(response, "text")
        'hi'
        >>> extract_value(response, "missing", default="n/a")
        'n/a'
    """
    # Try direct dictionary access
    if isinstance(response, dict) and key in response:
        return response[key]

    # Try attribute access
    if hasattr(response, key):
        return getattr(response, key)

    # Try data dictionary if it exists
    if hasattr(response, "data") and isinstance(response.data, dict) and key in response.data:
        return response.data[key]

    # Try to see if it's a nested structure
    if isinstance(response, dict):
        for _k, v in response.items():
            if isinstance(v, dict) and key in v:
                return v[key]

    # Return default if all else fails
    return default


__all__ = [
    # Base types
    "EmberModel",  # Base model for input/output types
    "Field",  # Field descriptor for validation constraints
    # Validators
    "field_validator",  # Field-level validation decorator
    "model_validator",  # Model-level validation decorator
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
