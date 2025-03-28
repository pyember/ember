"""Configuration loader module.

This module provides functions for loading configuration from various sources
and transforming it into a validated EmberConfig object.
"""

import os
import re
from typing import Any, Dict, List, Optional

from .exceptions import ConfigError
from .schema import EmberConfig


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
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
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
    # Debug print
    import logging

    logger = logging.getLogger("ember.core.config.loader")
    logger.debug(f"Resolving environment variables in config: {config}")

    if not isinstance(config, dict):
        return config

    result = {}

    for key, value in config.items():
        if isinstance(value, dict):
            logger.debug(f"Resolving env vars in nested dict at key {key}: {value}")
            result[key] = resolve_env_vars(value)
        elif isinstance(value, list):
            resolved_list = []
            for item in value:
                if isinstance(item, dict):
                    resolved_list.append(resolve_env_vars(item))
                elif isinstance(item, str) and "${" in item and "}" in item:
                    # Handle environment variable substitution in string items
                    pattern = r"\${([^}]+)}"
                    matches = re.findall(pattern, item)
                    result_value = item
                    for var_name in matches:
                        env_value = os.environ.get(var_name, "")
                        result_value = result_value.replace(
                            f"${{{var_name}}}", env_value
                        )
                    resolved_list.append(result_value)
                else:
                    resolved_list.append(item)
            result[key] = resolved_list
        elif isinstance(value, str) and "${" in value and "}" in value:
            # Simple pattern matching for environment variables
            pattern = r"\${([^}]+)}"
            matches = re.findall(pattern, value)

            if matches:
                logger.debug(
                    f"Found env var pattern in string at key {key}: {value}, matches: {matches}"
                )
                result_value = value
                for var_name in matches:
                    env_value = os.environ.get(var_name, "")
                    logger.debug(
                        f"Substituting env var {var_name} with value: '{env_value}'"
                    )
                    result_value = result_value.replace(f"${{{var_name}}}", env_value)
                result[key] = result_value
                logger.debug(f"After substitution: {result_value}")
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
        with open(path, "r") as f:
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
        "LOGGING_LEVEL": ["logging", "level"],
    }

    # Check for exact matches first
    if env_key in key_mappings:
        return key_mappings[env_key]

    # Default behavior: convert to lowercase and split by underscore
    path = env_key.lower().split("_")

    # Handle compound words that should stay together
    compound_words = [
        "auto_discover",
        "rate_limit",
        "api_key",
        "api_keys",
        "cost_input",
        "cost_output",
    ]

    # Check if any adjacent elements in path should be merged
    i = 0
    while i < len(path) - 1:
        combined = f"{path[i]}_{path[i+1]}"
        if combined in compound_words:
            path[i] = combined  # Replace first element with combined
            path.pop(i + 1)  # Remove second element
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
            env_key = key[len(prefix_upper) + 1 :]

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


def normalize_config_schema(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize configuration data to the latest schema format.

    Transforms legacy configuration formats to the current schema,
    handling field names and structure changes without modifying the
    schema classes themselves.

    Args:
        config_data: Raw configuration dictionary

    Returns:
        Normalized configuration ready for validation
    """
    result = config_data.copy()

    # Handle legacy registry paths
    if "model_registry" in result and "registry" not in result:
        # Map model_registry to registry namespace
        result["registry"] = result.pop("model_registry")

    # Process provider configurations
    if "registry" in result and "providers" in result["registry"]:
        providers = result["registry"]["providers"]
        for provider_name, provider_config in providers.items():
            # Handle models
            if "models" in provider_config:
                # Convert list format to dict format
                if isinstance(provider_config["models"], list):
                    models_dict = {}
                    for model in provider_config["models"]:
                        if isinstance(model, dict):
                            # Use ID as key, default to model name if no ID
                            model_id = model.get("id", model.get("name", "unknown"))

                            # Ensure required fields are present
                            if "provider" not in model:
                                model["provider"] = provider_name

                            models_dict[model_id] = model
                    provider_config["models"] = models_dict

                # Ensure each model has a provider field
                if isinstance(provider_config["models"], dict):
                    for model_id, model in provider_config["models"].items():
                        if isinstance(model, dict) and "provider" not in model:
                            model["provider"] = provider_name

                            # If provider is set but id doesn't contain provider prefix, add it
                            if "id" in model and ":" not in model["id"]:
                                model["id"] = f"{provider_name}:{model['id']}"

    return result


def load_config(
    file_path: Optional[str] = None, env_prefix: str = "EMBER"
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

        # Normalize config to current schema before validation
        config_data = normalize_config_schema(config_data)

        # Create and validate config object
        return EmberConfig.model_validate(config_data)

    except Exception as e:
        if isinstance(e, ConfigError):
            raise
        raise ConfigError(f"Failed to load configuration: {e}")
