"""Structured schemas describing Ember's configuration files."""

from typing import Any, Dict, Optional, TypedDict


class ProviderConfig(TypedDict, total=False):
    """Configuration options for a single provider entry."""

    api_key: Optional[str]
    base_url: Optional[str]
    organization_id: Optional[str]
    default_model: Optional[str]
    timeout: Optional[int]
    max_retries: Optional[int]


class ModelOverrideConfig(TypedDict, total=False):
    """Per-model override options applied after discovery merges."""

    description: Optional[str]
    pricing: Dict[str, Any]
    hidden: Optional[bool]
    aliases: list[str]
    capabilities: list[str]
    context_window: Optional[int]
    context_window_out: Optional[int]


class ModelsConfig(TypedDict, total=False):
    """Model-level generation defaults."""

    default: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    discovery_mode: str
    overrides: Dict[str, ModelOverrideConfig]


class LoggingConfig(TypedDict, total=False):
    """Logging preferences for Ember's runtime."""

    level: str
    format: str
    file: Optional[str]
    components: Dict[str, str]


class EmberConfig(TypedDict, total=False):
    """Top-level Ember configuration structure."""

    version: str
    models: ModelsConfig
    providers: Dict[str, ProviderConfig]
    logging: LoggingConfig
    data: Dict[str, Any]  # Dataset-specific config
    xcs: Dict[str, Any]  # XCS-specific config


class CredentialEntry(TypedDict):
    """Stored credential metadata."""

    api_key: str
    created_at: str
    last_used: Optional[str]


# Type alias for credentials file structure
CredentialsDict = Dict[str, CredentialEntry]
