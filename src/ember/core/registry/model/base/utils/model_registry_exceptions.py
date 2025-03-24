"""Model registry exceptions module.

This module provides re-exports of all model-related exceptions from ember.core.exceptions
for unified error handling throughout the Ember framework.
"""

# Re-export all model-related exceptions from the core exceptions module
from ember.core.exceptions import (
    InvalidArgumentError,
    InvalidPromptError,
    ModelDiscoveryError,
    ModelError,
    ModelNotFoundError,
    ModelProviderError,
    ModelRegistrationError,
    ProviderAPIError,
    ProviderConfigError,
    RegistryError,
    ValidationError,
)

__all__ = [
    "ModelError",
    "ModelProviderError",
    "ModelNotFoundError",
    "ProviderAPIError",
    "ProviderConfigError",
    "ModelDiscoveryError",
    "ModelRegistrationError",
    "InvalidPromptError",
    "InvalidArgumentError",
    "ValidationError",
    "RegistryError",
]
