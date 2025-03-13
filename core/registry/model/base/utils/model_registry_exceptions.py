"""Model registry exceptions module.

This module provides custom exceptions for model registry operations and re-exports
external exceptions from 'src.ember.core.exceptions' for unified error handling.
"""

# Re-export external exceptions for use in higher-level modules.
from ember.core.exceptions import (
    ProviderAPIError,
    ProviderConfigError,
    InvalidPromptError,
    ModelNotFoundError,
    RegistryError,
)

__all__ = [
    "ProviderAPIError",
    "ProviderConfigError",
    "InvalidPromptError",
    "ModelNotFoundError",
    "RegistryError",
    "ModelRegistrationError",
    "ModelDiscoveryError",
]


class ModelRegistrationError(Exception):
    """Exception raised when model registration fails.

    Attributes:
        model_name (str): Name of the model that failed registration.
        reason (str): Explanation detailing why the registration failed.
    """

    def __init__(self, *, model_name: str, reason: str) -> None:
        """Initializes a ModelRegistrationError with a detailed error message.

        Args:
            model_name (str): The name of the model.
            reason (str): The reason for the registration failure.
        """
        error_message: str = f"Failed to register model '{model_name}': {reason}"
        super().__init__(error_message)
        self.model_name: str = model_name
        self.reason: str = reason


class ModelDiscoveryError(Exception):
    """Exception raised when the model discovery process fails.

    Attributes:
        provider (str): Identifier of the provider for which discovery failed.
        reason (str): Explanation detailing why discovery failed.
    """

    def __init__(self, *, provider: str, reason: str) -> None:
        """Initializes a ModelDiscoveryError with a descriptive error message.

        Args:
            provider (str): The provider identifier.
            reason (str): The reason discovery failed.
        """
        error_message: str = f"Discovery failed for provider '{provider}': {reason}"
        super().__init__(error_message)
        self.provider: str = provider
        self.reason: str = reason
