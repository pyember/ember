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
