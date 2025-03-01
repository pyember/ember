"""
Type definitions for configuration objects.

This module provides type-safe definitions for configuration objects
used throughout the Ember system, replacing generic Dict[str, Any] with
more precise TypedDict and Protocol definitions.
"""

from typing import Dict, Any, Optional, Protocol, runtime_checkable, TypedDict, Union
from typing_extensions import NotRequired, Required

from .protocols import EmberTyped, TypeInfo


@runtime_checkable
class ConfigManager(Protocol):
    """
    Protocol defining the interface for configuration managers.

    This protocol ensures that any config manager can be used interchangeably
    as long as it provides the basic get/set operations.
    """

    def get(
        self, section: str, key: str, fallback: Optional[str] = None
    ) -> Optional[str]:
        """
        Retrieve a configuration value.

        Args:
            section: The configuration section
            key: The key within the section
            fallback: Default value if not found

        Returns:
            The configuration value or fallback
        """
        ...

    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            section: The configuration section
            key: The key within the section
            value: The value to set
        """
        ...


class ModelConfigDict(TypedDict, total=False):
    """
    Type-safe configuration for models.

    Replaces generic Dict[str, Any] with a structured TypedDict
    that documents the expected fields and their types.
    """

    openai_api_key: NotRequired[str]
    anthropic_api_key: NotRequired[str]
    google_api_key: NotRequired[str]
    model_defaults: NotRequired[Dict[str, Any]]


class EmberConfigDict(TypedDict, total=False):
    """
    Top-level configuration dictionary for Ember.

    Provides a structured view of the expected configuration sections.
    """

    models: Required[ModelConfigDict]
    logging: NotRequired[Dict[str, Any]]
    system: NotRequired[Dict[str, Any]]
