"""Configuration module for Ember's model registry.

This module provides functionality for loading YAML configuration files,
resolving environment variable placeholders, and constructing a strongly typed
EmberSettings object. The EmberSettings object is subsequently used to
initialize and configure the ModelRegistry.
"""

import logging
import os
import re
import yaml
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ember.core.exceptions import EmberError
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.registry.model_registry import ModelRegistry

logger: logging.Logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Removed Global Singletons.
# Previously declared:
#   GLOBAL_MODEL_REGISTRY: ModelRegistry = ModelRegistry()
#   GLOBAL_USAGE_SERVICE: UsageService = UsageService()
#   _INITIALIZED: bool = False


def deep_merge(*, base: Any, override: Any) -> Any:
    """Recursively merge two composite data structures.

    When both inputs are dictionaries, their keys are merged recursively.
    When both inputs are lists, they are concatenated.
    For other types, the override value replaces the base value.

    Args:
        base (Any): The base data structure.
        override (Any): The overriding data structure.

    Returns:
        Any: The merged data structure with the override applied.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged: Dict[Any, Any] = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], (dict, list)):
                merged[key] = deep_merge(base=merged[key], override=value)
            else:
                merged[key] = value
        return merged
    if isinstance(base, list) and isinstance(override, list):
        return base + override
    return override


def resolve_env_vars(*, data: Any) -> Any:
    """Recursively resolve environment variable placeholders in a data structure.

    Scans the provided data for string values formatted as '${VAR}' and replaces them
    with the corresponding environment variable's value (or an empty string if undefined).

    Args:
        data (Any): A nested data structure containing dictionaries, lists, or strings.

    Returns:
        Any: The data structure with environment variables resolved.
    """
    if isinstance(data, dict):
        return {key: resolve_env_vars(data=value) for key, value in data.items()}
    if isinstance(data, list):
        return [resolve_env_vars(data=item) for item in data]
    if isinstance(data, str):
        match: Optional[re.Match[str]] = re.fullmatch(r"\${([^}]+)}", data)
        if match:
            env_var: str = match.group(1)
            return os.environ.get(env_var, "")
        return data
    return data


class RegistryConfig(BaseModel):
    """Configuration settings for the Ember model registry.

    Attributes:
        auto_register (bool): If True, models will be auto-registered.
        auto_discover (bool): If True, models will be auto-discovered.
        included_configs (List[str]): File paths for additional configuration files.
        models (List[ModelInfo]): Locally defined model information objects.
    """

    auto_register: bool = False
    auto_discover: bool = False
    included_configs: List[str] = Field(default_factory=list)
    models: List[ModelInfo] = Field(default_factory=list)


class OtherSettings(BaseModel):
    """Miscellaneous Ember-level settings.

    Attributes:
        debug (Optional[bool]): Flag to enable debug mode; defaults to None.
        logging (Optional[Dict[str, Any]]): Logging configuration; defaults to None.
    """

    debug: Optional[bool] = None
    logging: Optional[Dict[str, Any]] = None


class EmberSettings(BaseSettings):
    """Top-level application settings for Ember.

    Settings are loaded from environment variables (.env) and merged with YAML
    configuration content.

    Attributes:
        model_config_path (str): Path to the model configuration YAML file.
        openai_api_key (Optional[str]): API key for OpenAI.
        anthropic_api_key (Optional[str]): API key for Anthropic.
        google_api_key (Optional[str]): API key for Google.
        registry (RegistryConfig): Configuration settings for the Ember model registry.
        other (OtherSettings): Additional miscellaneous settings.
        model_config (SettingsConfigDict): Pydantic settings configuration for environment variable loading.
    """

    model_config_path: str = Field(
        default_factory=lambda: os.path.join(os.path.dirname(__file__), "config.yaml")
    )
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    registry: RegistryConfig = RegistryConfig()
    other: OtherSettings = OtherSettings()
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", protected_namespaces=()
    )


def _initialize_model_registry(*, settings: EmberSettings) -> ModelRegistry:
    """Initialize a ModelRegistry using merged YAML and environment configuration.

    This function executes the following steps:
      1. Loads the config YAML from settings.model_config_path (if present).
      2. Optionally merges included files, etc. (for now, we just load the single file).
      3. Resolves environment variable placeholders within the config.
      4. Instantiates our EmberSettings and ModelRegistry.
      5. Optionally discovers remote models (if auto_discover is enabled) and merges them with local config.
      6. Registers all models into the registry.

    Args:
        settings (EmberSettings): An instance specifying configuration paths and registry preferences.

    Returns:
        ModelRegistry: A fully populated ModelRegistry instance with registered local and discovered models.

    Raises:
        EmberError: If configuration loading or processing fails.
    """
    try:
        merged_config: Dict[str, Any] = {}
        if os.path.isfile(settings.model_config_path):
            with open(settings.model_config_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
            merged_config = deep_merge(base=merged_config, override=file_config)
            logger.debug("Loaded config from %s", settings.model_config_path)
        else:
            logger.debug(
                "No file found at %s, using empty or custom config logic.",
                settings.model_config_path,
            )
    except Exception as exc:
        logger.exception(
            "Failed to handle config for '%s'.", settings.model_config_path
        )
        raise EmberError("Configuration loading error") from exc

    merged_config = resolve_env_vars(data=merged_config)
    logger.debug("Final merged config keys: %s", list(merged_config.keys()))

    final_settings: EmberSettings = EmberSettings(**merged_config)
    registry: ModelRegistry = ModelRegistry(logger=logger)

    discovered_models: Dict[str, ModelInfo] = {}
    if final_settings.registry.auto_discover:
        from ember.core.registry.model.base.registry.discovery import (
            ModelDiscoveryService,
        )

        discovery_service = ModelDiscoveryService(ttl=3600)
        discovered: Dict[str, Dict[str, Any]] = discovery_service.discover_models()
        discovered_models = discovery_service.merge_with_config(discovered=discovered)

    local_models: Dict[str, ModelInfo] = {
        model.id: model for model in final_settings.registry.models
    }
    discovered_models.update(local_models)

    for model_id, model_info in discovered_models.items():
        try:
            registry.register_model(model_info=model_info)
            logger.info("Registered model: %s", model_id)
        except Exception as exc:
            logger.error("Failed to register model '%s': %s", model_id, exc)

    logger.info("Model registry initialization complete.")
    return registry


def initialize_ember(
    config_path: Optional[str] = None,
    auto_register: bool = True,
    auto_discover: bool = True,
) -> ModelRegistry:
    """Initialize Ember's model registry via a unified configuration flow.

    This function is the recommended entry point for setting up Ember. It performs:
      - Loading and deep-merging of YAML configuration files (including additional included configs).
      - Resolution of environment variable placeholders.
      - Construction of an EmberSettings instance.
      - Instantiation of a new ModelRegistry.
      - Optional remote model discovery (if auto_discover is True).

    Example:
        registry = initialize_ember(
            config_path="./my_config.yaml",
            auto_register=True,
            auto_discover=True
        )

    Args:
        config_path (Optional[str]): Custom path to the main model configuration YAML file.
        auto_register (bool): If True, both local and discovered models are automatically registered.
        auto_discover (bool): If True, remote model discovery is performed and merged with local config.

    Returns:
        ModelRegistry: A fully populated ModelRegistry instance.

    Raises:
        EmberError: If configuration loading fails.
    """
    settings_obj: EmberSettings = EmberSettings()
    if config_path is not None:
        settings_obj.model_config_path = config_path

    settings_obj.registry.auto_register = auto_register
    settings_obj.registry.auto_discover = auto_discover

    registry_instance: ModelRegistry = _initialize_model_registry(settings=settings_obj)
    return registry_instance
