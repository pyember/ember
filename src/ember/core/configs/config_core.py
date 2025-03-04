"""Configuration Core Module

This module provides core functionality for loading, merging, and processing
configuration files in the Ember framework. It handles YAML configuration loading,
hierarchical configuration merging, and environment variable resolution.

Key features:
- Deep merging of nested configuration structures
- Environment variable interpolation in configuration values
- Support for configuration inclusion and inheritance
- Robust error handling for configuration loading

The module is designed to support a flexible configuration system where:
1. Base configurations can include and extend other configurations
2. Values can be overridden at multiple levels
3. Sensitive information can be stored in environment variables
4. Configuration can be adapted to different deployment environments

Usage example:
```python
# Load a configuration with inheritance
config = load_full_config(base_config_path="/path/to/config.yaml")

# Access configuration values
api_key = config.get("services", {}).get("model_api", {}).get("api_key")
```
"""

import os
import re
import yaml
import logging
from typing import Any, Dict, List

logger: logging.Logger = logging.getLogger(__name__)


def deep_merge(*, base: Any, override: Any) -> Any:
    """Recursively merge two composite data structures.

    This function takes two data structures and, if they are both dictionaries, it
    recursively merges them with values from the override taking precedence. If they are
    both lists, it concatenates them. In all other cases, the override value is returned.

    Args:
        base (Any): The base data structure.
        override (Any): The overriding data structure whose values take precedence.

    Returns:
        Any: The merged data structure.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged: Dict[Any, Any] = base.copy()
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(base=merged[key], override=value)
            else:
                merged[key] = value
        return merged
    if isinstance(base, list) and isinstance(override, list):
        return base + override
    return override


def resolve_env_vars(*, data: Any) -> Any:
    """Recursively resolve environment variable placeholders within a data structure.

    This function searches for string values matching the pattern '${VAR_NAME}' and replaces
    them with the corresponding environment variable. It recurses through dictionaries and lists
    to resolve environment variables in nested structures.

    Args:
        data (Any): The data in which to resolve environment variable placeholders. This may be
            a dictionary, list, or string.

    Returns:
        Any: The data with all environment variable placeholders resolved.
    """
    if isinstance(data, dict):
        return {key: resolve_env_vars(data=value) for key, value in data.items()}
    if isinstance(data, list):
        return [resolve_env_vars(data=item) for item in data]
    if isinstance(data, str):
        match = re.fullmatch(r"\${([^}]+)}", data)
        if match:
            env_var: str = match.group(1)
            return os.environ.get(env_var, "")
        return data
    return data


def load_full_config(*, base_config_path: str) -> Dict[str, Any]:
    """Load and merge a base YAML configuration with any included configurations.

    This function loads a YAML configuration file from the specified path. It then checks for any
    included configuration files listed under the key 'registry.included_configs' within the base
    configuration. Each included configuration is loaded and merged recursively using `deep_merge`.
    Lastly, the merged configuration is post-processed to resolve any environment variable placeholders.

    Args:
        base_config_path (str): The file system path to the base YAML configuration file.

    Raises:
        FileNotFoundError: If the base configuration file does not exist.

    Returns:
        Dict[str, Any]: The fully merged configuration dictionary with environment variables resolved.
    """
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(
            f"Model configuration file not found: {base_config_path}"
        )
    with open(base_config_path, "r", encoding="utf-8") as config_file:
        base_config: Dict[str, Any] = yaml.safe_load(config_file) or {}

    included_configs: List[str] = base_config.get("registry", {}).get(
        "included_configs", []
    )
    for include_path in included_configs:
        logger.info("Merging included config: %s", include_path)
        if os.path.exists(include_path):
            with open(include_path, "r", encoding="utf-8") as config_file:
                include_data: Dict[str, Any] = yaml.safe_load(config_file) or {}
            base_config = deep_merge(base=base_config, override=include_data)
        else:
            logger.warning("Included config not found: %s", include_path)

    return resolve_env_vars(data=base_config)
