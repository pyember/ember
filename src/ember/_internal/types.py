"""Typed surfaces that wrap Pydantic models for internal Ember usage.

Examples:
    >>> from ember._internal.types import EmberModel, Field
    >>> class UserInput(EmberModel):
    ...     text: str = Field(min_length=1, max_length=1000)
    ...     temperature: float = Field(ge=0.0, le=2.0, default=1.0)
"""

from pydantic import BaseModel
from pydantic import Field as PydanticField

# EmberModel is a simple alias for Pydantic's BaseModel, providing
# full validation features with zero overhead.
EmberModel = BaseModel

# Field provides validation constraints without exposing pydantic directly
# This abstraction allows us to change the underlying implementation later
Field = PydanticField

__all__ = ["EmberModel", "Field"]
