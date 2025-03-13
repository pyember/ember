"""Configuration providers module.

This module defines abstract and concrete provider classes for loading and saving 
configuration data from different sources such as files, environment variables, 
and more.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Generic, TypeVar, Optional
from pathlib import Path
import os

from .exceptions import ConfigError

T = TypeVar('T')  # Configuration data type


class ConfigProvider(Generic[T], ABC):
    """Abstract base class for configuration providers.
    
    Providers are responsible for loading raw configuration data from a specific source
    (files, environment variables, remote services, etc.) without validation.
    """
    
    @abstractmethod
    def load(self) -> T:
        """Load configuration from the source.
        
        Returns:
            T: Raw configuration data
        """
        pass
        
    @abstractmethod
    def save(self, config: T) -> None:
        """Save configuration to the source.
        
        Args:
            config: Configuration data to save
        """
        pass


class YamlFileProvider(ConfigProvider[Dict[str, Any]]):
    """YAML file-based configuration provider."""
    
    def __init__(self, file_path: str):
        """Initialize with a file path.
        
        Args:
            file_path: Path to the YAML configuration file
        """
        # Resolve environment variables in the path
        expanded_path = os.path.expandvars(file_path)
        self.file_path = Path(expanded_path).resolve()
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Dict[str, Any]: Parsed YAML content
            
        Raises:
            ConfigError: For parsing errors
        """
        try:
            import yaml
        except ImportError:
            raise ConfigError("PyYAML package is required for YAML configuration")
        
        try:
            if not self.file_path.exists():
                return {}
                
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML file {self.file_path}: {e}") from e
        except Exception as e:
            raise ConfigError(f"Error reading config file {self.file_path}: {e}") from e
    
    def save(self, config: Dict[str, Any]) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            
        Raises:
            ConfigError: If saving fails
        """
        try:
            import yaml
        except ImportError:
            raise ConfigError("PyYAML package is required for YAML configuration")
        
        try:
            # Ensure directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            raise ConfigError(f"Failed to save config to {self.file_path}: {e}") from e


class EnvironmentProvider(ConfigProvider[Dict[str, Any]]):
    """Environment variable configuration provider.
    
    Maps environment variables to a nested configuration structure using
    a prefix and delimiter pattern (e.g., EMBER_MODELS_API_KEY -> models.api_key).
    """
    
    def __init__(self, prefix: str = "EMBER", delimiter: str = "_"):
        """Initialize with prefix and delimiter.
        
        Args:
            prefix: Environment variable prefix to filter by
            delimiter: Character used to separate nested keys
        """
        self.prefix = prefix
        self.delimiter = delimiter
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from environment variables.
        
        Returns:
            Dict[str, Any]: Nested configuration dictionary
        """
        config = {}
        prefix_with_delim = f"{self.prefix}{self.delimiter}"
        
        for key, value in os.environ.items():
            if key.startswith(prefix_with_delim):
                # Remove prefix and split by delimiter
                path = key[len(prefix_with_delim):].lower().split(self.delimiter)
                
                # Build nested dictionary
                current = config
                for part in path[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set the final value
                current[path[-1]] = self._convert_value(value)
                
        return config
    
    def save(self, config: Dict[str, Any]) -> None:
        """Not implemented - cannot write to environment variables."""
        raise NotImplementedError("Cannot write to environment variables")
    
    def _convert_value(self, value: str) -> Any:
        """Convert string values to appropriate types.
        
        Args:
            value: String value from environment
            
        Returns:
            Any: Converted value (bool, int, float, or string)
        """
        # Convert boolean strings
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
            
        # Try numeric conversion
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            # Keep as string
            return value