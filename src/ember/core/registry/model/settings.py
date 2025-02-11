import logging
import os
import re
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .core.schemas.model_info import ModelInfo
from .core.services.usage_service import UsageService
from .model_registry import ModelRegistry
from ember.core.configs.config import load_full_config  # Consolidated config module
from ember.core.exceptions import EmberError  # Import centralized exception

logger: logging.Logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Removed Global Singletons.
# Previously declared:
# GLOBAL_MODEL_REGISTRY: ModelRegistry = ModelRegistry()
# GLOBAL_USAGE_SERVICE: UsageService = UsageService()
# _INITIALIZED: bool = False


def deep_merge(*, base: Any, override: Any) -> Any:
    """Recursively merge two composite data structures.

    Merges the overriding structure into the base structure. When both the base and
    override are dictionaries, their keys are merged recursively. When both are lists,
    the lists are concatenated. For any other types, the override value replaces the base value.

    Args:
        base (Any): The original data structure.
        override (Any): The overriding data structure.

    Returns:
        Any: The merged data structure with overrides applied.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged: Dict[Any, Any] = base.copy()
        for key, value in override.items():
            if key in merged:
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
    with the value of the corresponding environment variable.

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
        match = re.fullmatch(r"\${([^}]+)}", data)
        if match:
            env_var: str = match.group(1)
            return os.environ.get(env_var, "")
        return data
    return data


class RegistryConfig(BaseModel):
    """Configuration settings for the Ember model registry.

    Attributes:
        auto_register (bool): Indicates whether models should be auto-registered.
        auto_discover (bool): Indicates whether models should be auto-discovered.
        included_configs (List[str]): File paths for additional configuration files.
        models (List[ModelInfo]): Locally defined model information objects.
    """

    auto_register: bool = False
    auto_discover: bool = False
    included_configs: List[str] = Field(default_factory=list)
    models: List[ModelInfo] = Field(default_factory=list)


class OtherSettings(BaseModel):
    """Miscellaneous settings for the Ember application.

    Attributes:
        debug (Optional[bool]): Optional flag to enable debug mode.
        logging (Optional[Dict[str, Any]]): Optional logging configuration settings.
    """

    debug: Optional[bool] = None
    logging: Optional[Dict[str, Any]] = None


class EmberSettings(BaseSettings):
    """Application settings for Ember, including API keys and model registry configurations.

    Attributes:
        model_config_path (str): File path to the model configuration YAML file.
        openai_api_key (Optional[str]): API key for OpenAI.
        anthropic_api_key (Optional[str]): API key for Anthropic.
        google_api_key (Optional[str]): API key for Google.
        registry (RegistryConfig): Model registry configuration settings.
        other (OtherSettings): Additional miscellaneous settings.
        model_config (SettingsConfigDict): Pydantic settings configuration for loading from .env files.
    """

    model_config_path: str = Field(
        default_factory=lambda: os.path.join(os.path.dirname(__file__), "config.yaml")
    )
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    registry: RegistryConfig = RegistryConfig()
    other: OtherSettings = OtherSettings()
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env", extra="ignore", protected_namespaces=("settings_",)
    )


def initialize_model_registry(settings: EmberSettings) -> ModelRegistry:
    """
    Initialize a local model registry using the merged YAML configuration.
    This function reads the YAML configuration, optionally auto-discovers models,
    and registers all models into a new ModelRegistry instance.

    Returns:
        A freshly initialized ModelRegistry instance.
    """
    # 1) Load the full config.
    try:
        merged_config: Dict[str, Any] = load_full_config(
            base_config_path=settings.model_config_path
        )
    except Exception as e:
        logger.exception(
            "Failed to load full config from %s", settings.model_config_path
        )
        raise EmberError("Configuration loading error") from e

    final_settings: EmberSettings = EmberSettings(**merged_config)

    # 2) Create a new model registry instance.
    registry = ModelRegistry(logger=logger)

    # 3) Auto-discover models if enabled.
    discovered_models: Dict[str, ModelInfo] = {}
    if final_settings.registry.auto_discover:
        from .discovery_service import ModelDiscoveryService

        discovery_service = ModelDiscoveryService(ttl=3600)
        discovered: Dict[str, Dict[str, Any]] = discovery_service.discover_models()
        discovered_models = discovery_service.merge_with_config(discovered=discovered)

    # 4) Merge locally defined models (local overrides).
    local_models: Dict[str, ModelInfo] = {
        model.model_id: model for model in final_settings.registry.models
    }
    discovered_models.update(local_models)

    # 5) Register all models into the registry.
    for model_id, model_info in discovered_models.items():
        try:
            registry.register_model(model_info=model_info)
            logger.info("Registered model: %s", model_id)
        except Exception as err:
            logger.error("Failed to register model %s: %s", model_id, err)

    logger.info("Model registry initialization complete.")
    return registry
