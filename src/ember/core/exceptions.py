class EmberError(Exception):
    """Base class for all custom exceptions in the Ember library."""

    pass


class ProviderAPIError(EmberError):
    """Raised when an external provider API call fails."""

    pass


class ProviderConfigError(EmberError):
    """Raised when there is an issue with provider configuration."""

    pass


class InvalidPromptError(EmberError):
    """Raised when a prompt is invalid or malformed."""

    pass


class ModelNotFoundError(EmberError):
    """Raised when a requested model is not found in the registry."""

    pass


class RegistryError(EmberError):
    """Base class for errors related to registry operations."""

    pass


class ValidationError(EmberError):
    """Raised when input validation fails."""

    pass


class ConfigurationError(EmberError):
    """Raised when there's a configuration error."""

    pass


class MissingLMModuleError(EmberError):
    """Raised when the expected LM module is missing from an operator."""

    pass


__all__ = [
    "EmberError",
    "ProviderAPIError",
    "ProviderConfigError",
    "InvalidPromptError",
    "ModelNotFoundError",
    "RegistryError",
    "ValidationError",
    "ConfigurationError",
    "MissingLMModuleError",
]
