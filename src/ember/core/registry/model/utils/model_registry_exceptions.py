from src.ember.core.exceptions import (
    ProviderAPIError,
    ProviderConfigError,
    InvalidPromptError,
    ModelNotFoundError,
    RegistryError,
)

"""Custom exceptions for model registry operations."""


class ModelRegistrationError(Exception):
    """Raised when model registration fails."""

    def __init__(self, model_name: str, reason: str):
        super().__init__(f"Failed to register model '{model_name}': {reason}")
        self.model_name = model_name
        self.reason = reason


class ModelDiscoveryError(Exception):
    """Raised when model discovery process fails."""

    def __init__(self, provider: str, reason: str):
        super().__init__(f"Discovery failed for provider '{provider}': {reason}")
        self.provider = provider
        self.reason = reason


__all__ = ["ModelRegistrationError", "ModelDiscoveryError"]
