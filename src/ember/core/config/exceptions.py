"""Configuration exception module.

This module defines exception types specific to the configuration system.
"""

from ember.core.exceptions import EmberError


class ConfigError(EmberError):
    """Exception raised for configuration errors.
    
    This includes errors such as:
    - Invalid configuration format
    - Missing required configuration
    - Configuration validation failures
    - File access errors
    """
    pass