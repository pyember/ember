import configparser
import logging
import os
from pathlib import Path
from typing import Any, Optional


# -------------------------------------------------------------------
# 1) Configuration Manager
# -------------------------------------------------------------------
class ConfigManager:
    """Manages configuration using configparser.

    This class loads and writes configuration settings from a file.
    If the specified configuration file does not exist, it initializes
    default settings and saves them to disk.

    Attributes:
        _config (configparser.ConfigParser): The internal configuration parser.
        _config_path (Path): The filesystem path to the configuration file.
    """

    def __init__(self, config_filename: Optional[str] = None) -> None:
        """Initializes a ConfigManager instance.

        Args:
            config_filename (Optional[str]): An optional explicit path to the
                configuration file. If provided, this overrides the 'ember_CONFIG'
                environment variable. Defaults to None, which results in using
                'config.ini' if neither is set.
        """
        self._config: configparser.ConfigParser = configparser.ConfigParser()
        env_path: str = (
            os.environ.get("ember_CONFIG") or config_filename or "config.ini"
        )
        self._config_path: Path = Path(env_path).resolve()
        self._load_or_init_config()

    def _load_or_init_config(self) -> None:
        """Loads the configuration file if it exists or initializes default settings.

        If the configuration file exists at the resolved path, this method reads it;
        otherwise, it initializes a default configuration and persists it.
        Subsequently, it ensures that required sections and keys are present.
        """
        if self._config_path.exists():
            self._config.read(self._config_path)
        else:
            self._init_defaults()
            self._save()

        # Ensure that the 'models' section exists.
        if "models" not in self._config:
            self._config["models"] = {}

        # Ensure that the 'openai_api_key' is present in the 'models' section.
        if "openai_api_key" not in self._config["models"]:
            self._config["models"]["openai_api_key"] = ""

        # Re-save configuration if defaults were added.
        self._save()

    def _init_defaults(self) -> None:
        """Initializes default configuration settings."""
        self._config["models"] = {"openai_api_key": ""}

    def _save(self) -> None:
        """Saves the current configuration to disk."""
        with self._config_path.open(mode="w") as configfile:
            self._config.write(configfile)

    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """Retrieves a configuration value.

        Args:
            section (str): The section from which to retrieve the value.
            key (str): The key whose value is to be retrieved.
            fallback (Any, optional): A default value if the key is not found.
                Defaults to None.

        Returns:
            Any: The configuration value, or the fallback if the key is not found.
        """
        return self._config.get(section=section, option=key, fallback=fallback)

    def set(self, section: str, key: str, value: Any) -> None:
        """Sets a configuration value and persists the change.

        Args:
            section (str): The section in which to set the value.
            key (str): The key for the configuration entry.
            value (Any): The value to store.
        """
        if not self._config.has_section(section):
            self._config.add_section(section)
        self._config.set(section=section, option=key, value=str(value))
        self._save()


# Global, shared configuration object.
CONFIG: ConfigManager = ConfigManager()


# -------------------------------------------------------------------
# 2) API Key Initialization
# -------------------------------------------------------------------
def initialize_api_keys() -> None:
    """Initializes API keys from environment variables.

    This function loads or overrides the model API keys from environment variables,
    updating the configuration accordingly. It logs warnings if any expected API key
    is missing.
    """
    # OpenAI API Key initialization.
    openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        CONFIG.set(section="models", key="openai_api_key", value=openai_api_key)
    elif not CONFIG.get(section="models", key="openai_api_key"):
        logging.warning(
            "WARNING: OpenAI API key not found in environment variables or configuration file."
        )

    # Anthropic API Key initialization.
    anthropic_api_key: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        CONFIG.set(section="models", key="anthropic_api_key", value=anthropic_api_key)
    elif not CONFIG.get(section="models", key="anthropic_api_key"):
        logging.warning(
            "WARNING: Anthropic API key not found in environment variables or configuration file."
        )

    # Google API Key initialization.
    google_api_key: Optional[str] = os.environ.get("GOOGLE_API_KEY")
    if google_api_key:
        CONFIG.set(section="models", key="google_api_key", value=google_api_key)
    elif not CONFIG.get(section="models", key="google_api_key"):
        logging.warning(
            "WARNING: Google API key not found in environment variables or configuration file."
        )


# -------------------------------------------------------------------
# 3) Model Registration Logic
# -------------------------------------------------------------------
def auto_register_known_models(registry: Any) -> None:
    """Auto-registers known models into a model registry.

    This function pre-populates the provided registry with default models if they are
    not already registered. For demonstration purposes, it auto-registers an OpenAI GPT-4o model.

    Args:
        registry (Any): An object supporting the methods get_model() and register_model().
    """
    from src.ember.registry.model.schemas.model_info import ModelInfo
    from src.ember.registry.model.schemas.provider_info import ProviderInfo
    from src.ember.registry.model.schemas.cost import ModelCost, RateLimit

    known_model_id: str = "openai:gpt-4o"
    if registry.get_model(model_id=known_model_id) is None:
        logging.info("Auto-registering known model: %s", known_model_id)
        cost: ModelCost = ModelCost(
            input_cost_per_thousand=0.03, output_cost_per_thousand=0.06
        )
        rate: RateLimit = RateLimit(tokens_per_minute=80000, requests_per_minute=5000)
        openai_api_key: str = CONFIG.get(
            section="models", key="openai_api_key", fallback="DUMMY_KEY"
        )
        model_info: ModelInfo = ModelInfo(
            model_id=known_model_id,
            model_name="gpt-4o",
            cost=cost,
            rate_limit=rate,
            provider=ProviderInfo(name="OpenAI", default_api_key=openai_api_key),
            api_key=openai_api_key,
        )
        registry.register_model(model_info=model_info)


# -------------------------------------------------------------------
# 4) Initialization Entry Point
# -------------------------------------------------------------------
def initialize_system(registry: Optional[Any] = None) -> None:
    """Initializes system configuration and model registry.

    This function loads API keys from the environment and, if a model registry is provided,
    auto-registers known models into it.

    Args:
        registry (Optional[Any]): An optional model registry supporting auto-registration.
            Defaults to None.
    """
    initialize_api_keys()
    if registry is not None:
        auto_register_known_models(registry=registry)


# Automatically initialize API keys upon module import.
initialize_api_keys()
