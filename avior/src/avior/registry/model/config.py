import os
import logging
import yaml
import re
from typing import Dict, Any, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field

from src.avior.registry.model.registry.model_registry import ModelRegistry
from src.avior.registry.model.services.usage_service import UsageService
from src.avior.registry.model.schemas.model_info import ModelInfo
from src.avior.registry.model.schemas.provider_info import ProviderInfo
from src.avior.registry.model.schemas.cost import ModelCost, RateLimit

logger = logging.getLogger(__name__)

GLOBAL_MODEL_REGISTRY = ModelRegistry()
GLOBAL_USAGE_SERVICE = UsageService()

_INITIALIZED = False


def deep_merge(base: Any, override: Any) -> Any:
    """
    Recursively merges `override` into `base`, returning `base`.
    - If both `base` and `override` are dicts, we merge keys recursively.
    - If both are lists, we append the override elements to base.
    - Otherwise, we overwrite `base` with `override`.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        for k, v in override.items():
            base[k] = deep_merge(base.get(k), v)
        return base
    elif isinstance(base, list) and isinstance(override, list):
        base.extend(override)
        return base
    else:
        return override


###############################################################################
# NEW FUNCTION: Expand any string of the form "${ENV_VAR}" from your environment.
###############################################################################
def resolve_env_vars(data: Any) -> Any:
    """
    Recursively walks `data`. If a string exactly matches pattern "${SOMETHING}",
    we replace it with `os.environ.get("SOMETHING", "")`.
    """
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = resolve_env_vars(v)
        return data
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = resolve_env_vars(data[i])
        return data
    elif isinstance(data, str):
        # Only replace if the entire string is a single environment placeholder.
        match = re.match(r"^\${([^}]+)}$", data)
        if match:
            env_name = match.group(1)
            return os.environ.get(env_name, "")
        else:
            return data
    else:
        return data


class RegistryConfig(BaseModel):
    auto_register: bool = False
    included_configs: List[str] = Field(default_factory=list)
    models: List[ModelInfo] = Field(default_factory=list)


class OtherSettings(BaseModel):
    debug: bool | None = None
    logging: dict | None = None
    # ... any other custom fields you expect in config.yaml ...


class AviorSettings(BaseSettings):
    """
    Example Pydantic-based settings.
    Reads environment variables + your 'config.yaml' by default.
    """

    # path to the main config.yaml
    model_config_path: str = os.path.join(os.path.dirname(__file__), "config.yaml")

    # environment-based overrides for provider keys, etc.
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    # Adding a nested 'registry' field that will come from config.yaml
    registry: RegistryConfig = RegistryConfig()

    # Add a field to hold other settings
    other: OtherSettings = Field(default_factory=OtherSettings)

    # pydantic settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    def load_model_config(self) -> Dict[str, Any]:
        """
        Loads the main config.yaml specified by model_config_path.
        Returns a Python dict of all config data.
        """
        if not os.path.exists(self.model_config_path):
            raise FileNotFoundError(
                f"Model configuration file not found: {self.model_config_path}"
            )

        with open(self.model_config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError("Top-level YAML structure must be a dictionary.")
        return data


def initialize_global_registry() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        logger.info("Global registry already initialized. Skipping.")
        return

    logger.info("Entering initialize_global_registry()")

    # 1) Load environment settings
    env_settings = AviorSettings()
    logger.debug(
        "Environment-based settings (before YAML merge): %s", env_settings.model_dump()
    )

    # 2) Load the base config.yaml into a dict
    yaml_data = env_settings.load_model_config()
    logger.debug("Loaded config.yaml data: %s", yaml_data)

    # 3) Merge environment + YAML
    merged_config_dict = deep_merge(env_settings.model_dump(), yaml_data)
    logger.debug("After merging environment + config.yaml: %s", merged_config_dict)

    ###############################################################################
    # Perform environment-variable expansion BEFORE we parse into AviorSettings:
    ###############################################################################
    merged_config_dict = resolve_env_vars(merged_config_dict)

    # Parse into AviorSettings
    main_config = AviorSettings(**merged_config_dict)
    logger.debug("Parsed main_config (before includes): %s", main_config.model_dump())

    logger.info("Merging included config files...")
    for i_cfg in main_config.registry.included_configs:
        logger.info(f"Merging included config: {i_cfg}")
        if os.path.exists(i_cfg):
            with open(i_cfg, "r", encoding="utf-8") as f:
                sub_data = yaml.safe_load(f) or {}
                logger.debug("sub_data from %s = %s", i_cfg, sub_data)
                merged_config_dict = deep_merge(merged_config_dict, sub_data)
                ###############################################################################
                # Re-expand any environment placeholders after merging each sub-config:
                ###############################################################################
                merged_config_dict = resolve_env_vars(merged_config_dict)
                logger.debug("merged_config_dict so far: %s", merged_config_dict)
        else:
            logger.warning(f"Included config path does not exist: {i_cfg}")

    # Final parse
    main_config = AviorSettings(**merged_config_dict)
    logger.debug("Final main_config after includes: %s", main_config.model_dump())

    if main_config.registry.auto_register and main_config.registry.models:
        logger.info("Auto-registering models from the final config:")
        for m in main_config.registry.models:
            logger.info(
                f"  -> Registering model_id='{m.model_id}', provider='{m.provider.name}'"
            )
            GLOBAL_MODEL_REGISTRY.register_model(m)
    else:
        logger.warning("No models found to auto-register or 'auto_register' is false.")

    logger.info("Global registry initialization complete.")
    _INITIALIZED = True
