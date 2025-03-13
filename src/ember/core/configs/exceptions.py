"""Configuration exceptions module.

This module defines exception classes specific to configuration errors in the Ember framework.
"""

from ember.core.exceptions import EmberError


class ConfigError(EmberError):
    """Exception raised for errors in the configuration system.
    
    This includes errors such as:
    - Configuration file not found
    - Invalid configuration format
    - Validation errors 
    - Missing required configuration
    - Permission errors when saving configuration
    """
    pass