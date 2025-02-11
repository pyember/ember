"""
Ember Models Main Module

This module provides a unified initialization function to configure the Ember model module
environment. It loads and merges YAML configurations (with environment variable
resolution), auto-registers local models, optionally discovers remote models, and
returns a fully-configured ModelService instance. This is the primary entry point
for initializing Ember models.
"""

from typing import Optional

from ember.core.registry.model.config.settings import initialize_ember
from ember.core.registry.model.services.model_service import ModelService
from ember.core.registry.model.services.usage_service import UsageService


def init(
    *,
    config_path: Optional[str] = None,
    auto_register: bool = True,
    auto_discover: bool = True,
    usage_tracking: bool = False,
) -> ModelService:
    """Initializes Ember's model registry and returns a fully configured ModelService.

    This function performs the following steps:
      1. Loads and merges YAML configuration files, including resolving environment variables.
      2. Auto-registers local models as specified in the configuration.
      3. Optionally auto-discovers remote models.
      4. Optionally enables usage tracking via a UsageService.

    Args:
        config_path (Optional[str]): Custom path to the primary YAML configuration file.
            Defaults to None.
        auto_register (bool): Whether to automatically register local models per the configuration.
            Defaults to True.
        auto_discover (bool): Whether to automatically discover remote models. Defaults to True.
        usage_tracking (bool): If True, enables usage tracking through a UsageService.
            Defaults to False.

    Returns:
        ModelService: A fully initialized ModelService instance, ready for model invocation.
    """
    # Initialize the registry using the unified settings flow.
    registry = initialize_ember(
        config_path=config_path,
        auto_register=auto_register,
        auto_discover=auto_discover,
    )

    # Conditionally create a UsageService if usage tracking is enabled.
    usage_service: Optional[UsageService] = UsageService() if usage_tracking else None

    # Return the ModelService with explicit named parameters.
    return ModelService(registry=registry, usage_service=usage_service)
