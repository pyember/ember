"""Configuration Management Module

This module provides the primary configuration management functionality for Ember,
including API key management, configuration file handling, and model auto-registration.

The configuration system is designed around several key principles:
1. Configuration can come from multiple sources (env vars, config files, code)
2. Sensitive data like API keys should be handled securely
3. Common configurations should have sensible defaults
4. The system should be friendly for both development and production

Key components:
- ConfigManager: Core class for managing INI-style configuration
- API key initialization: Logic for loading API keys from environment variables
- Model auto-registration: Convenience functionality to register common models

The module supports dependency injection to make testing easier and configurations
more flexible across different environments.

Usage example:
```python
# Basic configuration setup
config_manager = ConfigManager()
initialize_api_keys(config_manager)

# Create a model registry with auto-registered models
registry = ModelRegistry()
auto_register_known_models(registry, config_manager)

# Access configuration
openai_key = config_manager.get("models", "openai_api_key")
```
"""

import configparser
import logging
import os
from pathlib import Path
from typing import Any, Optional

from ember.core.registry.model.base.registry.model_registry import ModelRegistry


class EmberError(Exception):
    """Base class for all custom exceptions in the Ember library.

    Attributes:
        Exception: Base exception for Ember-related errors.
    """

    pass


class ConfigManager:
    """Manages configuration using configparser with dependency injection support.

    Attributes:
        _config (configparser.ConfigParser): Internal configuration parser.
        _config_path (Path): Filesystem path to the configuration file.
        _logger (logging.Logger): Logger instance used for configuration events.
    """

    def __init__(
        self,
        config_filename: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize a ConfigManager instance.

        Args:
            config_filename (Optional[str]): Explicit path to the configuration file.
                Overrides the 'ember_CONFIG' environment variable if provided.
                Defaults to None, which results in 'config.ini' if not set.
            logger (Optional[logging.Logger]): Logger instance to use. Defaults to None,
                in which case a default logger is created.

        Returns:
            None
        """
        self._config: configparser.ConfigParser = configparser.ConfigParser()
        self._logger: logging.Logger = (
            logger if logger is not None else logging.getLogger(self.__class__.__name__)
        )
        env_path: str = (
            os.environ.get("ember_CONFIG") or config_filename or "config.ini"
        )
        self._config_path: Path = Path(env_path).resolve()
        self._load_or_init_config()

    @property
    def logger(self) -> logging.Logger:
        """Return the internal logger instance.

        Returns:
            logging.Logger: The logger used by this ConfigManager.
        """
        return self._logger

    def _load_or_init_config(self) -> None:
        """Load configuration from disk or initialize default settings.

        If the configuration file exists at the resolved path, it is read. Otherwise, the
        default configuration is initialized and persisted. Additionally, mandatory sections
        and keys are ensured.

        Returns:
            None
        """
        if self._config_path.exists():
            self._config.read(self._config_path)
        else:
            self._init_defaults()
            self._save()

        if "models" not in self._config:
            self._config["models"] = {}

        if "openai_api_key" not in self._config["models"]:
            self._config["models"]["openai_api_key"] = ""

        self._save()

    def _init_defaults(self) -> None:
        """Initialize default configuration settings.

        Returns:
            None
        """
        self._config["models"] = {"openai_api_key": ""}

    def _save(self) -> None:
        """Persist the current configuration to disk.

        Returns:
            None
        """
        with self._config_path.open(mode="w") as configfile:
            self._config.write(configfile)

    def get(
        self, section: str, key: str, fallback: Optional[str] = None
    ) -> Optional[str]:
        """Retrieve a configuration value.

        Args:
            section (str): The configuration section to query.
            key (str): The key within the section whose value is desired.
            fallback (Optional[str]): The default value if the key is not found. Defaults to None.

        Returns:
            Optional[str]: The configuration value if it exists; otherwise, the fallback.
        """
        return self._config.get(section=section, option=key, fallback=fallback)

    def set(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value and persist the change.

        Args:
            section (str): The configuration section in which to set the value.
            key (str): The key to set within the section.
            value (Any): The value to store (will be converted to a string).

        Returns:
            None
        """
        if not self._config.has_section(section):
            self._config.add_section(section)
        self._config.set(section=section, option=key, value=str(value))
        self._save()


def initialize_api_keys(
    config_manager: ConfigManager, logger: Optional[logging.Logger] = None
) -> None:
    """Initialize API keys from environment variables into the configuration.

    This function retrieves API keys from environment variables and updates the
    configuration accordingly. If an API key is missing in both the environment and the
    configuration, a warning is logged.

    Args:
        config_manager (ConfigManager): The configuration manager to update.
        logger (Optional[logging.Logger]): Optional logger for warnings. If None, the
            logger from config_manager is used.

    Returns:
        None
    """
    local_logger: logging.Logger = (
        logger if logger is not None else config_manager.logger
    )

    openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        config_manager.set(section="models", key="openai_api_key", value=openai_api_key)
    elif not config_manager.get(section="models", key="openai_api_key"):
        local_logger.warning(
            "WARNING: OpenAI API key not found in environment variables or configuration file."
        )

    anthropic_api_key: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        config_manager.set(
            section="models", key="anthropic_api_key", value=anthropic_api_key
        )
    elif not config_manager.get(section="models", key="anthropic_api_key"):
        local_logger.warning(
            "WARNING: Anthropic API key not found in environment variables or configuration file."
        )

    google_api_key: Optional[str] = os.environ.get("GOOGLE_API_KEY")
    if google_api_key:
        config_manager.set(section="models", key="google_api_key", value=google_api_key)
    elif not config_manager.get(section="models", key="google_api_key"):
        local_logger.warning(
            "WARNING: Google API key not found in environment variables or configuration file."
        )


def auto_register_known_models(
    registry: ModelRegistry, config_manager: ConfigManager
) -> None:
    """Automatically register known models in the model registry.

    Checks whether a known model is present in the registry; if not, it registers
    the model using configuration settings.

    Args:
        registry (ModelRegistry): The model registry to update.
        config_manager (ConfigManager): The configuration manager providing model settings.

    Returns:
        None
    """
    local_logger: logging.Logger = config_manager.logger
    from ember.core.registry.model.base.schemas.model_info import ModelInfo
    from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
    from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit

    known_model_id: str = "openai:gpt-4o"
    if registry.get_model(model_id=known_model_id) is None:
        local_logger.info("Auto-registering known model: %s", known_model_id)
        cost: ModelCost = ModelCost(
            input_cost_per_thousand=0.03, output_cost_per_thousand=0.06
        )
        rate: RateLimit = RateLimit(tokens_per_minute=80000, requests_per_minute=5000)
        openai_api_key: str = (
            config_manager.get(
                section="models", key="openai_api_key", fallback="DUMMY_KEY"
            )
            or "DUMMY_KEY"
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


def initialize_system(
    app_context: Optional["EmberAppContext"] = None,
    registry: Optional[ModelRegistry] = None,
) -> None:
    """Initialize system configuration and model registry.

    Depending on the presence of an application context, this function initializes API keys and,
    if provided, auto-registers known models using the configuration manager from the application context.
    If no application context is provided, a temporary configuration manager is used and a warning
    is logged.

    Args:
        app_context (Optional["EmberAppContext"]): The application context containing configuration.
        registry (Optional[ModelRegistry]): The model registry for auto-registration of models.

    Returns:
        None
    """
    if app_context is not None:
        initialize_api_keys(config_manager=app_context.config_manager)
        if registry is not None:
            auto_register_known_models(
                registry=registry, config_manager=app_context.config_manager
            )
    else:
        temp_config: ConfigManager = ConfigManager()
        temp_config.logger.warning(
            "Initializing without AppContext - prefer create_ember_app()"
        )
        initialize_api_keys(config_manager=temp_config)
        if registry is not None:
            auto_register_known_models(registry=registry, config_manager=temp_config)
