"""Configuration loader module.

This module provides functions for loading configuration from various sources
and transforming it into a validated EmberConfig object.
"""

from typing import Dict, Any, Optional, List
import os
import re
from pathlib import Path

from .schema import EmberConfig
from .exceptions import ConfigError


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary with values that override the base
        
    Returns:
        Merged dictionary where override values take precedence
    """
    result = base.copy()
    
    for key, value in override.items():
        if (key in result and 
            isinstance(result[key], dict) and 
            isinstance(value, dict)):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
            
    return result


def resolve_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Replace ${VAR} patterns with environment variables.
    
    Args:
        config: Dictionary containing configuration values
        
    Returns:
        Configuration with environment variables resolved
    """
    if not isinstance(config, dict):
        return config
        
    result = {}
    
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = resolve_env_vars(value)
        elif isinstance(value, list):
            resolved_list = []
            for item in value:
                if isinstance(item, dict):
                    resolved_list.append(resolve_env_vars(item))
                elif isinstance(item, str) and "${" in item and "}" in item:
                    # Handle environment variable substitution in string items
                    pattern = r'\${([^}]+)}'
                    matches = re.findall(pattern, item)
                    result_value = item
                    for var_name in matches:
                        env_value = os.environ.get(var_name, "")
                        result_value = result_value.replace(f"${{{var_name}}}", env_value)
                    resolved_list.append(result_value)
                else:
                    resolved_list.append(item)
            result[key] = resolved_list
        elif isinstance(value, str) and "${" in value and "}" in value:
            # Simple pattern matching for environment variables
            pattern = r'\${([^}]+)}'
            matches = re.findall(pattern, value)
            
            if matches:
                result_value = value
                for var_name in matches:
                    env_value = os.environ.get(var_name, "")
                    result_value = result_value.replace(f"${{{var_name}}}", env_value)
                result[key] = result_value
            else:
                result[key] = value
        else:
            result[key] = value
            
    return result


def load_yaml_file(path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Dictionary containing configuration from YAML
        
    Raises:
        ConfigError: If file cannot be read or parsed
    """
    try:
        import yaml
    except ImportError:
        raise ConfigError("PyYAML is required for YAML configuration")
        
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}")
    except Exception as e:
        raise ConfigError(f"Error reading {path}: {e}")


def _normalize_env_key(env_key: str) -> List[str]:
    """Normalize environment variable key to configuration path.
    
    Args:
        env_key: Environment variable key without prefix (e.g., "REGISTRY_AUTO_DISCOVER")
        
    Returns:
        List of path segments (e.g., ["registry", "auto_discover"])
    """
    # Custom mapping for well-known keys
    key_mappings = {
        "REGISTRY_AUTO_DISCOVER": ["registry", "auto_discover"],
        "LOGGING_LEVEL": ["logging", "level"]
    }
    
    # Check for exact matches first
    if env_key in key_mappings:
        return key_mappings[env_key]
    
    # Default behavior: convert to lowercase and split by underscore
    path = env_key.lower().split("_")
    
    # Handle compound words that should stay together
    compound_words = ["auto_discover", "rate_limit", "api_key", "api_keys", "cost_input", "cost_output"]
    
    # Check if any adjacent elements in path should be merged
    i = 0
    while i < len(path) - 1:
        combined = f"{path[i]}_{path[i+1]}"
        if combined in compound_words:
            path[i] = combined  # Replace first element with combined
            path.pop(i+1)       # Remove second element
        else:
            i += 1
    
    return path

def load_from_env(prefix: str = "EMBER") -> Dict[str, Any]:
    """Load configuration from environment variables with given prefix.
    
    Args:
        prefix: Prefix for environment variables to consider
        
    Returns:
        Dictionary containing configuration from environment
    """
    result = {}
    prefix_upper = prefix.upper()
    
    for key, value in os.environ.items():
        if key.startswith(f"{prefix_upper}_"):
            # Get the key without prefix
            env_key = key[len(prefix_upper) + 1:]
            
            # Normalize to config path
            path = _normalize_env_key(env_key)
            
            # Convert value to appropriate type
            if value.lower() in ("true", "yes", "1"):
                typed_value = True
            elif value.lower() in ("false", "no", "0"):
                typed_value = False
            elif value.isdigit():
                typed_value = int(value)
            elif value.replace(".", "", 1).isdigit():
                typed_value = float(value)
            else:
                typed_value = value
                
            # Build nested dictionary
            current = result
            for part in path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
                
            # Set value at final path
            current[path[-1]] = typed_value
            
    return result


def load_config(
    file_path: Optional[str] = None,
    env_prefix: str = "EMBER"
) -> EmberConfig:
    """Load EmberConfig from file and environment.
    
    Args:
        config_path: Path to config file (defaults to EMBER_CONFIG from env or "config.yaml")
        env_prefix: Prefix for environment variables
        
    Returns:
        Validated EmberConfig instance
        
    Raises:
        ConfigError: On loading or validation failure
    """
    try:
        # Determine config path
        path = file_path or os.environ.get(f"{env_prefix}_CONFIG", "config.yaml")
        
        # Start with default empty config
        config_data: Dict[str, Any] = {}
        
        # Load from file if it exists
        if os.path.exists(path):
            file_config = load_yaml_file(path)
            config_data = merge_dicts(config_data, file_config)
        
        # Load from environment (overrides file)
        env_config = load_from_env(env_prefix)
        if env_config:
            config_data = merge_dicts(config_data, env_config)
        
        # Resolve environment variables in strings
        config_data = resolve_env_vars(config_data)
        
        # Create and validate config object
        return EmberConfig.model_validate(config_data)
        
    except Exception as e:
        if isinstance(e, ConfigError):
            raise
        raise ConfigError(f"Failed to load configuration: {e}")