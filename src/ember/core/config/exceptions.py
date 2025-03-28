"""Configuration exception module.

This module re-exports exceptions from the core exceptions module to maintain
backward compatibility for any code importing from here.
"""

from ember.core.exceptions import (
    ConfigError,
    ConfigFileError,
    ConfigValidationError,
    ConfigValueError,
    MissingConfigError,
)

__all__ = [
    "ConfigError",
    "ConfigFileError",
    "ConfigValidationError",
    "ConfigValueError",
    "MissingConfigError",
]
