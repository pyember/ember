"""Configuration Manager Module.

This module provides a simple configuration management system for the Ember framework.
It handles loading, validating, and accessing configuration values.
"""

import logging
from threading import RLock
from typing import Any, Optional

from .exceptions import ConfigError
from .loader import load_config
from .schema import EmberConfig


class ConfigManager:
    """
    Configuration manager for Ember.

    This class handles loading, reloading, and accessing configuration values.
    It provides thread-safe access to configuration and manages API keys.
    """

    def __init__(
        self, config_path: Optional[str] = None, logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to configuration file
            logger: Logger for configuration events
        """
        self._lock = RLock()
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._config_path = config_path
        self._config = None  # Will be loaded on first access or explicit load() call

    def load(self) -> EmberConfig:
        """
        Load configuration from file and environment.

        Returns:
            Validated EmberConfig instance

        Raises:
            ConfigError: On loading or validation failure
        """
        with self._lock:
            try:
                self._logger.debug("Loading configuration...")
                config = load_config(file_path=self._config_path)
                self._logger.debug("Configuration loaded successfully")
                return config
            except Exception as e:
                self._logger.error(f"Error loading configuration: {e}")
                raise ConfigError(f"Failed to load configuration: {e}")

    def reload(self) -> EmberConfig:
        """
        Reload configuration from sources.

        Returns:
            Newly loaded EmberConfig instance
        """
        with self._lock:
            self._config = self.load()
            return self._config

    def get_config(self) -> EmberConfig:
        """
        Get the current configuration, loading if needed.

        Returns:
            Current EmberConfig instance
        """
        with self._lock:
            if self._config is None:
                self._config = self.load()
            return self._config

    def set_provider_api_key(self, provider_name: str, api_key: str) -> None:
        """
        Set API key for a specific provider.

        Args:
            provider_name: Provider identifier (e.g., "openai")
            api_key: API key to set
        """
        with self._lock:
            # Ensure config is loaded
            if self._config is None:
                self._config = self.load()

            if provider_name not in self._config.registry.providers:
                self._config.registry.providers[provider_name] = {}

            # Create providers.{provider}.api_keys.default.key
            if not hasattr(self._config.registry.providers[provider_name], "api_keys"):
                self._config.registry.providers[provider_name].api_keys = {}

            if "default" not in self._config.registry.providers[provider_name].api_keys:
                self._config.registry.providers[provider_name].api_keys["default"] = {}

            self._config.registry.providers[provider_name].api_keys["default"][
                "key"
            ] = api_key
            self._logger.debug(f"Set API key for provider {provider_name}")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by section and key.

        Args:
            section: Configuration section
            key: Key within the section
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        with self._lock:
            # Ensure config is loaded
            if self._config is None:
                self._config = self.load()

            try:
                if hasattr(self._config, section):
                    section_obj = getattr(self._config, section)
                    if hasattr(section_obj, key):
                        return getattr(section_obj, key)
            except (AttributeError, TypeError):
                pass

            return default


def create_config_manager(
    config_path: Optional[str] = None, logger: Optional[logging.Logger] = None
) -> ConfigManager:
    """
    Create and initialize a configuration manager.

    Args:
        config_path: Path to configuration file
        logger: Logger for configuration events

    Returns:
        Initialized ConfigManager instance
    """
    return ConfigManager(config_path=config_path, logger=logger)
