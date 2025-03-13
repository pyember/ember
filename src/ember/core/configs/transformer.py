"""Configuration transformer module.

This module provides classes and functions for transforming raw configuration
data before validation, such as resolving environment variables and deep merging.
"""

import os
import re
from typing import Dict, Any, List, Callable, TypeVar, Generic

T = TypeVar('T')


class ConfigTransformer(Generic[T]):
    """Transforms raw configuration data before validation.
    
    ConfigTransformers apply a series of transformations to raw configuration
    data, such as environment variable resolution, deep merging, etc.
    """
    
    def __init__(self, transformations: List[Callable[[T], T]] = None):
        """Initialize with a list of transformation functions.
        
        Args:
            transformations: Functions that transform config data
        """
        self.transformations = transformations or []
    
    def transform(self, config: T) -> T:
        """Apply all transformations to the configuration.
        
        Args:
            config: Raw configuration data
            
        Returns:
            T: Transformed configuration data
        """
        result = config
        for transform in self.transformations:
            result = transform(result)
        return result
    
    def add_transformation(self, transformation: Callable[[T], T]) -> None:
        """Add a transformation to the pipeline.
        
        Args:
            transformation: Function that transforms config data
        """
        self.transformations.append(transformation)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries.
    
    When both inputs are dictionaries, their keys are merged recursively.
    When both inputs are lists, they are concatenated.
    For other types, the override value replaces the base value.
    
    Args:
        base: Base dictionary
        override: Dictionary with overriding values
        
    Returns:
        Dict[str, Any]: Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            result[key] = result[key] + value
        else:
            result[key] = value
            
    return result


def resolve_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively resolve environment variable references.
    
    Replaces ${VAR_NAME} patterns with environment variable values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict[str, Any]: Configuration with resolved variables
    """
    def _resolve(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_resolve(item) for item in value]
        elif isinstance(value, str):
            # Match ${VAR_NAME} pattern
            matches = re.findall(r'\${([^}]+)}', value)
            if matches:
                result = value
                for match in matches:
                    env_value = os.environ.get(match, '')
                    result = result.replace(f'${{{match}}}', env_value)
                return result
            return value
        else:
            return value
            
    return _resolve(config)


def add_default_values(config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Add default values to a configuration dictionary.
    
    Only adds default values for keys that don't exist in the original config.
    
    Args:
        config: Original configuration
        defaults: Default values to add
        
    Returns:
        Dict[str, Any]: Configuration with defaults added
    """
    result = config.copy()
    
    for key, value in defaults.items():
        if key not in result:
            result[key] = value
        elif isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = add_default_values(result[key], value)
            
    return result