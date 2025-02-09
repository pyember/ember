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

logger: logging.Logger = logging.getLogger(__name__)

# Global singletons.
GLOBAL_MODEL_REGISTRY: ModelRegistry = ModelRegistry()
GLOBAL_USAGE_SERVICE: UsageService = UsageService()
_INITIALIZED: bool = False


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


def load_full_config(*, base_config_path: str) -> Dict[str, Any]:
    """Load and merge the base YAML configuration with additional included configuration files.

    Reads the primary YAML configuration from the specified path and then merges it
    with any supplementary configurations defined under the 'registry' section. Finally,
    it resolves any environment variable placeholders within the merged configuration.

    Args:
        base_config_path (str): The file path to the base YAML configuration.

    Returns:
        Dict[str, Any]: The merged configuration dictionary with resolved environment variables.

    Raises:
        FileNotFoundError: If the base configuration file does not exist.
    """
    if not os.path.exists(path=base_config_path):
        raise FileNotFoundError(
            f"Model configuration file not found: {base_config_path}"
        )
    with open(file=base_config_path, mode="r", encoding="utf-8") as config_file:
        base_config: Dict[str, Any] = yaml.safe_load(config_file) or {}  # type: ignore

    included_configs: List[str] = base_config.get("registry", {}).get(
        "included_configs", []
    )
    for include_path in included_configs:
        logger.info("Merging included config: %s", include_path)
        if os.path.exists(path=include_path):
            with open(file=include_path, mode="r", encoding="utf-8") as include_file:
                include_data: Dict[str, Any] = yaml.safe_load(include_file) or {}  # type: ignore
            base_config = deep_merge(base=base_config, override=include_data)
        else:
            logger.warning("Included config not found: %s", include_path)

    return resolve_env_vars(data=base_config)


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


def initialize_global_registry() -> None:
    """Initialize the global model registry using configuration and environment variables.

    This procedure performs the following steps:
      1. Loads initial settings from the environment and .env file.
      2. Loads and merges the YAML configuration from the specified file along with any included files.
      3. Optionally auto-discovers models if enabled.
      4. Merges locally defined models which override any discovered configurations.
      5. Registers all models into the global model registry.

    Raises:
        Exception: Propagates any exceptions encountered during model registration.
    """
    global _INITIALIZED
    if _INITIALIZED:
        logger.info("Global registry already initialized; skipping re-initialization.")
        return

    # Step 1: Load initial settings from the environment and .env file.
    env_settings: EmberSettings = EmberSettings()

    # Step 2: Load and merge the YAML configuration.
    merged_config: Dict[str, Any] = load_full_config(
        base_config_path=env_settings.model_config_path
    )
    final_settings: EmberSettings = EmberSettings(**merged_config)

    # Step 3: Optionally auto-discover models.
    discovered_models: Dict[str, ModelInfo] = {}
    if final_settings.registry.auto_discover:
        from .discovery_service import ModelDiscoveryService

        discovery_service = ModelDiscoveryService(ttl=3600)
        discovered: Dict[str, ModelInfo] = discovery_service.discover_models()
        discovered_models = discovery_service.merge_with_config(discovery=discovered)

    # Step 4: Merge locally defined models (local models override discovered ones).
    local_models: Dict[str, ModelInfo] = {
        model.model_id: model for model in final_settings.registry.models
    }
    discovered_models.update(local_models)

    # Step 5: Register all models in the global registry.
    for model_id, model_info in discovered_models.items():
        try:
            GLOBAL_MODEL_REGISTRY.register_model(model_info=model_info)
            logger.info("Registered model: %s", model_id)
        except Exception as err:
            logger.error("Failed to register model %s: %s", model_id, err)

    logger.info("Global registry initialization complete.")
    _INITIALIZED = True
