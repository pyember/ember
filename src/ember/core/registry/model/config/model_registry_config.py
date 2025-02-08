import logging
import os
import re
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ember.core.registry.model.config.model_registry import ModelRegistry
from ember.core.registry.model.core.schemas.model_info import ModelInfo
from ember.core.registry.model.core.services.usage_service import UsageService

logger: logging.Logger = logging.getLogger(__name__)

GLOBAL_MODEL_REGISTRY: ModelRegistry = ModelRegistry()
GLOBAL_USAGE_SERVICE: UsageService = UsageService()

_INITIALIZED: bool = False


def deep_merge(base: Any, override: Any) -> Any:
    """Recursively merges two data structures.

    Merges `override` into `base` using the following rules:
      - If both `base` and `override` are dicts, merge keys recursively.
      - If both are lists, extend `base` with the elements of `override`.
      - Otherwise, return `override`, overwriting `base`.

    Args:
        base: The original data structure.
        override: The data structure providing overriding values.

    Returns:
        The merged data structure.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        for key, value in override.items():
            base[key] = deep_merge(base.get(key), value)
        return base
    elif isinstance(base, list) and isinstance(override, list):
        base.extend(override)
        return base
    else:
        return override


def resolve_env_vars(data: Any) -> Any:
    """Recursively resolve environment variable placeholders in the given data.

    This function traverses `data` and if it finds a string that exactly matches
    the pattern "${ENV_VAR}", it replaces it with the value of the environment
    variable named ENV_VAR or an empty string if the variable is not set.

    Args:
        data: A data structure (dict, list, or str) potentially containing placeholders.

    Returns:
        The data structure with environment variables resolved.
    """
    if isinstance(data, dict):
        return {key: resolve_env_vars(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [resolve_env_vars(item) for item in data]
    elif isinstance(data, str):
        match = re.fullmatch(r"\${([^}]+)}", data)
        if match:
            env_var: str = match.group(1)
            return os.environ.get(env_var, "")
        return data
    else:
        return data


class RegistryConfig(BaseModel):
    """Configuration settings for the model registry.

    Attributes:
        auto_register: If True, models specified in this config will be auto-registered.
        auto_discover: If True, models will be auto-discovered from external providers.
        included_configs: Paths to additional YAML configuration files.
        models: A list of local model definitions.
    """

    auto_register: bool = False
    auto_discover: bool = False
    included_configs: List[str] = Field(default_factory=list)
    models: List[ModelInfo] = Field(default_factory=list)


class OtherSettings(BaseModel):
    """Additional settings defined in the configuration file.

    Attributes:
        debug: Enables debug mode when set to True.
        logging: A dictionary containing logging configurations.
    """

    debug: Optional[bool] = None
    logging: Optional[Dict[str, Any]] = None


class EmberSettings(BaseSettings):
    """Application settings that combine environment variables with YAML configuration.

    Attributes:
        model_config_path: The file path to the primary YAML configuration file.
        openai_api_key: API key for accessing OpenAI services.
        anthropic_api_key: API key for accessing Anthropic services.
        google_api_key: API key for accessing Google services.
        registry: Nested registry-specific configurations.
        other: Additional miscellaneous settings.
    """

    model_config_path: str = os.path.join(os.path.dirname(__file__), "config.yaml")
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    registry: RegistryConfig = RegistryConfig()
    other: OtherSettings = Field(default_factory=OtherSettings)

    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    def load_model_config(self) -> Dict[str, Any]:
        """Loads the main YAML configuration file.

        Reads the file specified by `model_config_path` and returns its contents as a dictionary.

        Returns:
            A dictionary representing the YAML configuration.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the YAML file's top-level structure is not a dictionary.
        """
        if not os.path.exists(self.model_config_path):
            raise FileNotFoundError(
                f"Model configuration file not found: {self.model_config_path}"
            )

        with open(self.model_config_path, "r", encoding="utf-8") as file:
            config_data: Any = yaml.safe_load(file) or {}
            if not isinstance(config_data, dict):
                raise ValueError("Top-level YAML structure must be a dictionary.")
        return config_data


def initialize_global_registry() -> None:
    """Initializes the global model registry using configuration settings.

    This function:
      1. Loads settings from environment variables.
      2. Reads and merges the main YAML configuration file.
      3. Processes additional included configuration files.
      4. Resolves environment variable placeholders.
      5. Auto-discovers or auto-registers models based on configuration flags.

    Subsequent calls will have no effect once initialization is complete.
    """
    global _INITIALIZED
    if _INITIALIZED:
        logger.info("Global registry already initialized. Skipping initialization.")
        return

    logger.info("Starting global registry initialization.")

    # 1) Load environment settings.
    settings: EmberSettings = EmberSettings()
    logger.debug("Environment settings (pre-YAML merge): %s", settings.model_dump())

    # 2) Load the base YAML configuration.
    yaml_config: Dict[str, Any] = settings.load_model_config()
    logger.debug("Loaded YAML configuration: %s", yaml_config)

    # 3) Merge the environment settings with the YAML configuration.
    merged_config: Dict[str, Any] = deep_merge(settings.model_dump(), yaml_config)
    logger.debug("Configuration after merge: %s", merged_config)

    # 4) Resolve any environment variable placeholders.
    merged_config = resolve_env_vars(merged_config)

    # Parse the merged configuration.
    main_settings: EmberSettings = EmberSettings(**merged_config)
    logger.debug(
        "Parsed settings before processing included configs: %s",
        main_settings.model_dump(),
    )

    logger.info("Processing included configuration files.")
    for include_path in main_settings.registry.included_configs:
        logger.info("Merging included config: %s", include_path)
        if os.path.exists(include_path):
            with open(include_path, "r", encoding="utf-8") as include_file:
                include_data: Any = yaml.safe_load(include_file) or {}
                logger.debug("Configuration from %s: %s", include_path, include_data)
                merged_config = deep_merge(merged_config, include_data)
                merged_config = resolve_env_vars(merged_config)
                logger.debug(
                    "Merged configuration after %s: %s", include_path, merged_config
                )
        else:
            logger.warning("Included configuration file not found: %s", include_path)

    # Final parse after processing included configs.
    final_settings: EmberSettings = EmberSettings(**merged_config)
    logger.debug("Final merged settings: %s", final_settings.model_dump())

    # 5) Auto-discover or auto-register models.
    if final_settings.registry.auto_discover:
        from ember.core.registry.model.config.discovery_service import ModelDiscoveryService

        discovery_service: ModelDiscoveryService = ModelDiscoveryService(ttl=3600)
        discovered_models: Dict[str, Dict[str, Any]] = (
            discovery_service.discover_models()
        )
        merged_model_infos: Dict[str, ModelInfo] = discovery_service.merge_with_config(
            discovered_models
        )
        for model_info in merged_model_infos.values():
            try:
                logger.info(
                    "Auto-registering discovered model: %s", model_info.model_id
                )
                GLOBAL_MODEL_REGISTRY.register_model(model_info=model_info)
            except Exception as error:
                logger.error(
                    "Failed to register discovered model %s: %s",
                    model_info.model_id,
                    error,
                )
    elif final_settings.registry.auto_register and final_settings.registry.models:
        logger.info("Auto-registering models from configuration.")
        for model_conf in final_settings.registry.models:
            logger.info(
                "Registering model with model_id='%s', provider='%s'",
                model_conf.model_id,
                model_conf.provider.name,
            )
            GLOBAL_MODEL_REGISTRY.register_model(model_info=model_conf)
    else:
        logger.warning(
            "No models available for registration; either auto_register is disabled or no models are provided."
        )

    logger.info("Global registry initialization complete.")
    _INITIALIZED = True
