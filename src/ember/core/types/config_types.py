"""
Type definitions for configuration objects.

This module provides type-safe definitions for configuration objects
used throughout the Ember system, replacing generic Dict[str, Any] with
more precise TypedDict and Protocol definitions.
"""

from typing import (
    Any,
    Dict,
    List,
    Protocol,
    TypedDict,
    runtime_checkable,
)

from typing_extensions import NotRequired, Required


@runtime_checkable
class ConfigManager(Protocol):
    """
    Protocol defining the interface for configuration managers.

    This protocol ensures that any config manager can be used interchangeably
    as long as it provides these core operations.
    """

    def get_config(self) -> Any:
        """
        Get the current configuration.

        Returns:
            Current EmberConfig instance
        """
        ...

    def load(self) -> Any:
        """
        Load configuration from sources.

        Returns:
            Loaded EmberConfig instance
        """
        ...

    def reload(self) -> Any:
        """
        Reload configuration from sources.

        Returns:
            Updated EmberConfig instance
        """
        ...

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value.

        Args:
            section: The configuration section
            key: The key within the section
            default: Default value if not found

        Returns:
            The configuration value or default
        """
        ...

    def set_provider_api_key(self, provider_name: str, api_key: str) -> None:
        """
        Set API key for a specific provider.

        Args:
            provider_name: Provider identifier (e.g., "openai")
            api_key: API key to set
        """
        ...


class ApiKeyDict(TypedDict, total=False):
    """Type-safe configuration for API keys."""

    key: str
    org_id: NotRequired[str]
    endpoint: NotRequired[str]


class ModelCostDict(TypedDict):
    """Type-safe configuration for model costs."""

    input_cost_per_thousand: float
    output_cost_per_thousand: float


class RateLimitDict(TypedDict, total=False):
    """Type-safe configuration for rate limits."""

    tokens_per_minute: int
    requests_per_minute: int


class ModelDict(TypedDict):
    """Type-safe configuration for models."""

    id: str
    name: str
    provider: str
    cost: ModelCostDict
    rate_limit: NotRequired[RateLimitDict]
    parameters: NotRequired[Dict[str, Any]]


# Backward compatibility alias
ModelConfigDict = ModelDict


class ProviderDict(TypedDict, total=False):
    """Type-safe configuration for providers."""

    enabled: bool
    api_keys: Dict[str, ApiKeyDict]
    models: List[ModelDict]
    base_url: NotRequired[str]
    timeout: NotRequired[float]
    max_retries: NotRequired[int]


class RegistryDict(TypedDict, total=False):
    """Type-safe configuration for the registry."""

    auto_discover: bool
    auto_register: NotRequired[bool]
    providers: Dict[str, ProviderDict]


class LoggingDict(TypedDict, total=False):
    """Type-safe configuration for logging."""

    level: str
    format: NotRequired[str]
    file: NotRequired[str]


class EmberConfigDict(TypedDict, total=False):
    """
    Top-level configuration dictionary for Ember.

    Provides a structured view of the expected configuration sections.
    """

    registry: Required[RegistryDict]
    logging: NotRequired[LoggingDict]
    data_paths: NotRequired[Dict[str, str]]
