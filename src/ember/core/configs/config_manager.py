"""Configuration manager module.

This module provides the unified configuration management system for Ember,
handling loading, transforming, validating, and accessing configuration from
multiple sources with thread-safety guarantees.
"""

from typing import Dict, Any, List, Optional, Type, TypeVar, Generic, cast
from pathlib import Path
import logging
import threading
import os

from pydantic import BaseModel, ValidationError

from .providers import ConfigProvider, YamlFileProvider, EnvironmentProvider
from .transformer import ConfigTransformer, deep_merge, resolve_env_vars, add_default_values
from .exceptions import ConfigError
from .schema import EmberConfig, DEFAULT_CONFIG

T = TypeVar('T', bound=BaseModel)


class ConfigManager(Generic[T]):
    """Thread-safe configuration manager with multi-source support.
    
    ConfigManager orchestrates loading, transforming, validating, and
    providing access to configuration from multiple sources with
    thread-safety guarantees.
    
    Type Parameters:
        T: Pydantic model type for configuration validation
    """
    
    def __init__(
        self, 
        schema_class: Type[T], 
        providers: List[ConfigProvider] = None,
        transformers: List[ConfigTransformer] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize ConfigManager with sources and transformers.
        
        Args:
            schema_class: Pydantic model class for validation
            providers: List of configuration providers (files, env vars, etc.)
            transformers: List of configuration transformers
            logger: Logger instance
        """
        self._schema_class = schema_class
        self._providers = providers or []
        self._transformers = transformers or []
        self._config: Optional[T] = None
        self._lock = threading.RLock()
        self._logger = logger or logging.getLogger(self.__class__.__name__)
    
    def load(self) -> T:
        """Load and validate configuration from all sources.
        
        Returns:
            T: Validated configuration object
            
        Raises:
            ConfigError: If loading or validation fails
        """
        with self._lock:
            try:
                # Load from all providers
                configs: List[Dict[str, Any]] = []
                for provider in self._providers:
                    try:
                        provider_config = provider.load()
                        configs.append(cast(Dict[str, Any], provider_config))
                        self._logger.debug(
                            f"Loaded config from {provider.__class__.__name__}"
                        )
                    except Exception as e:
                        self._logger.warning(
                            f"Error loading from provider {provider.__class__.__name__}: {e}"
                        )
                
                # Start with default configuration
                merged_config = DEFAULT_CONFIG.copy()
                
                # Merge configs (later providers override earlier ones)
                for config in configs:
                    merged_config = deep_merge(merged_config, config)
                
                # Apply transformations
                for transformer in self._transformers:
                    merged_config = transformer.transform(merged_config)
                
                # Validate using schema
                validated_config = self._schema_class(**merged_config)
                self._config = validated_config
                
                self._logger.debug("Configuration loaded and validated successfully")
                return validated_config
                
            except ValidationError as e:
                error_msg = "Configuration validation failed: " + str(e)
                self._logger.error(error_msg)
                raise ConfigError(error_msg) from e
            except Exception as e:
                error_msg = f"Error loading configuration: {e}"
                self._logger.error(error_msg)
                raise ConfigError(error_msg) from e
    
    def get_config(self) -> T:
        """Get the current configuration.
        
        Returns:
            T: Validated configuration object
            
        Raises:
            ConfigError: If configuration hasn't been loaded
        """
        with self._lock:
            if self._config is None:
                return self.load()
            return self._config
    
    def save(self, provider_index: int = 0) -> None:
        """Save the current configuration to a provider.
        
        Args:
            provider_index: Index of the provider to save to (default: first provider)
            
        Raises:
            ConfigError: If saving fails or no providers exist
        """
        with self._lock:
            if not self._providers:
                raise ConfigError("No configuration providers available for saving")
                
            if self._config is None:
                raise ConfigError("No configuration loaded to save")
                
            try:
                provider = self._providers[provider_index]
                # Convert Pydantic model to dict
                config_dict = self._config.model_dump()
                provider.save(config_dict)
                self._logger.info(
                    f"Configuration saved using {provider.__class__.__name__}"
                )
            except IndexError:
                raise ConfigError(f"Provider index {provider_index} out of range")
            except Exception as e:
                raise ConfigError(f"Failed to save configuration: {e}") from e
    
    def reload(self) -> T:
        """Force reload configuration from all sources.
        
        Returns:
            T: Reloaded configuration object
        """
        with self._lock:
            self._config = None
            return self.load()
    
    def get(self, *path: str, default: Any = None) -> Any:
        """Get a configuration value by path.
        
        Args:
            *path: Path segments to the configuration value
            default: Default value if path doesn't exist
            
        Returns:
            Any: Configuration value or default
        """
        with self._lock:
            config = self.get_config()
            current = config.model_dump()
            
            for segment in path:
                if isinstance(current, dict) and segment in current:
                    current = current[segment]
                else:
                    return default
                    
            return current
    
    def set_provider_api_key(self, provider_name: str, api_key: str) -> None:
        """Set the API key for a provider in the configuration.
        
        Args:
            provider_name: Name of the provider (e.g., "openai")
            api_key: API key to set
            
        Raises:
            ConfigError: If provider doesn't exist in configuration
        """
        with self._lock:
            config = self.get_config()
            config_dict = config.model_dump()
            
            if (
                "model_registry" not in config_dict or
                "providers" not in config_dict["model_registry"] or
                provider_name not in config_dict["model_registry"]["providers"]
            ):
                # Add the provider if it doesn't exist
                if "model_registry" not in config_dict:
                    config_dict["model_registry"] = {}
                if "providers" not in config_dict["model_registry"]:
                    config_dict["model_registry"]["providers"] = {}
                    
                config_dict["model_registry"]["providers"][provider_name] = {
                    "enabled": True,
                    "api_keys": {"default": {"key": api_key}}
                }
            else:
                # Update existing provider
                provider = config_dict["model_registry"]["providers"][provider_name]
                if "api_keys" not in provider:
                    provider["api_keys"] = {}
                provider["api_keys"]["default"] = {"key": api_key}
            
            # Validate and update config
            self._config = self._schema_class(**config_dict)
            
            # Save the updated configuration
            if self._providers:
                self.save()


def create_default_config_manager(
    config_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> ConfigManager[EmberConfig]:
    """Create a default configuration manager for Ember.
    
    Args:
        config_path: Path to configuration file (default: env var or './config.yaml')
        logger: Logger instance
        
    Returns:
        ConfigManager[EmberConfig]: Configured manager instance
    """
    # Determine config path
    file_path = config_path or os.environ.get('EMBER_CONFIG', './config.yaml')
    
    # Configure logger
    if logger is None:
        logger = logging.getLogger("EmberConfig")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    # Create providers (environment overrides file)
    providers = [
        YamlFileProvider(file_path),
        EnvironmentProvider(prefix="EMBER")
    ]
    
    # Create transformers
    env_transformer = ConfigTransformer()
    env_transformer.add_transformation(resolve_env_vars)
    
    # Create and return manager
    return ConfigManager(
        schema_class=EmberConfig,
        providers=providers,
        transformers=[env_transformer],
        logger=logger
    )