"""
Ember Models Main Module

This module provides a unified initialization function to configure the Ember model module
environment. It loads and merges YAML configurations (with environment variable
resolution), auto-registers local models, optionally discovers remote models, and
returns a fully-configured ModelService instance. This is the primary entry point
for initializing Ember models.
"""

from __future__ import annotations

# Lazy imports to prevent circular dependencies
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ember.core.registry.model import ModelRegistry, ModelService
    from src.ember.core.registry.model import initialize_ember

from typing import Optional

from .core.registry.model import initialize_ember, ModelRegistry, ModelService
from .core.registry.model.services.usage_service import UsageService

__version__ = "0.1.0"
__all__ = ["ModelRegistry", "ModelService", "initialize_ember"]

# Package version and metadata
_PACKAGE_METADATA = {
    "name": "ember",
    "version": __version__,
    "description": "Compound AI Systems framework for Network of Network (NON) construction.",
}


def __getattr__(name: str) -> object:
    """Lazy load main components using absolute imports."""
    if name == "ModelRegistry":
        from src.ember.core.registry.model.registry.model_registry import ModelRegistry

        return ModelRegistry
    if name == "ModelService":
        from src.ember.core.registry.model.services.model_service import ModelService

        return ModelService
    if name == "initialize_ember":
        from src.ember.core.registry.model import initialize_ember

        return initialize_ember
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
