import configparser
import logging
import os
from pathlib import Path
from typing import Any, Optional

# -------------------------------------------------------------------
# 1) Configuration Manager
# -------------------------------------------------------------------
class ConfigManager:
    """
    Manages configuration via configparser, optionally loading from an
    environment-specified file. Facilitates reading/writing config settings
    in a consolidated, easy-to-use style.
    """

    def __init__(self, config_filename: Optional[str] = None):
        """
        :param config_filename: If provided, overrides the environment variable AVIOR_CONFIG.
        """
        self._config = configparser.ConfigParser()
        env_path = os.environ.get("AVIOR_CONFIG") or config_filename or "config.ini"
        self._config_path = Path(env_path).resolve()
        self._load_or_init_config()

    def _load_or_init_config(self) -> None:
        """Loads the config file if present; otherwise initializes defaults and saves."""
        if self._config_path.exists():
            self._config.read(self._config_path)
        else:
            self._init_defaults()
            self._save()

        # Ensure at least the 'models' section exists
        if "models" not in self._config:
            self._config["models"] = {}
        # Ensure openai_api_key is present
        if "openai_api_key" not in self._config["models"]:
            self._config["models"]["openai_api_key"] = ""

        # Re-save if we added any new defaults
        self._save()

    def _init_defaults(self) -> None:
        """Defines minimal default structure if config doesn't exist at all."""
        self._config["models"] = {
            "openai_api_key": "",
        }

    def _save(self) -> None:
        """Persists the current config state to disk."""
        with self._config_path.open("w") as configfile:
            self._config.write(configfile)

    # ---------------------------
    # Public API
    # ---------------------------
    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """
        Retrieves a configuration value with an optional fallback if not present.
        """
        return self._config.get(section, key, fallback=fallback)

    def set(self, section: str, key: str, value: Any) -> None:
        """
        Sets a configuration value in memory, then writes it to disk.
        """
        if not self._config.has_section(section):
            self._config.add_section(section)
        self._config.set(section, key, str(value))
        self._save()


# Global, shared config object
CONFIG = ConfigManager()

# -------------------------------------------------------------------
# 2) API Key Initialization
# -------------------------------------------------------------------
def initialize_api_keys() -> None:
    """
    Loads or overrides model API keys from environment variables.
    Logs warnings if keys are missing.
    """
    # 1) OpenAI
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        CONFIG.set("models", "openai_api_key", openai_api_key)
    elif not CONFIG.get("models", "openai_api_key"):
        print("WARNING: OpenAI API key not found in environment variables or config file")

    # 2) Anthropic
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        CONFIG.set("models", "anthropic_api_key", anthropic_api_key)
    elif not CONFIG.get("models", "anthropic_api_key"):
        print("WARNING: Anthropic API key not found in environment variables or config file")

    # 3) Google
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key:
        CONFIG.set("models", "google_api_key", google_api_key)
    elif not CONFIG.get("models", "google_api_key"):
        print("WARNING: Google API key not found in environment variables or config file")


# -------------------------------------------------------------------
# 3) Model Registration Logic
# -------------------------------------------------------------------
def auto_register_known_models(registry):
    """
    Pre-populates the ModelRegistry with known models if not already registered.
    If you have a dynamic or consolidated ModelEnum, you can iterate over it.
    Example:
        from .model_enum import ModelEnum
        for member in ModelEnum:
            model_id = member.value
            # build a minimal ModelInfo and call registry.register_model(...)
    """

    from src.avior.registry.model.schemas.model_info import ModelInfo
    from src.avior.registry.model.schemas.provider_info import ProviderInfo
    from src.avior.registry.model.schemas.cost import ModelCost, RateLimit

    # For demonstration, let's auto-register an OpenAI GPT-4o model if not present
    # Real usage: you'd loop over an enum or any other known model IDs
    known_model_id = "openai:gpt-4o"
    if registry.get_model(known_model_id) is None:
        import logging
        logging.info(f"Auto-registering known model: {known_model_id}")
        # Basic cost/rate-limit placeholders
        cost = ModelCost(input_cost_per_thousand=0.03, output_cost_per_thousand=0.06)
        rate = RateLimit(tokens_per_minute=80000, requests_per_minute=5000)
        openai_api_key = CONFIG.get("models", "openai_api_key", fallback="DUMMY_KEY")

        # Minimal ModelInfo
        model_info = ModelInfo(
            model_id=known_model_id,
            model_name="gpt-4o",
            cost=cost,
            rate_limit=rate,
            provider=ProviderInfo(name="OpenAI", default_api_key=openai_api_key),
            api_key=openai_api_key,
        )
        registry.register_model(model_info)


# -------------------------------------------------------------------
# 4) Initialization Entry Point
# -------------------------------------------------------------------
def initialize_system(registry=None) -> None:
    """
    Top-level initializer that:
      1. Ensures API keys are in config
      2. Optionally auto-registers known models in the given registry
    """
    initialize_api_keys()

    if registry:
        auto_register_known_models(registry)


# Automatically run basic config + key initialization at import time
initialize_api_keys()