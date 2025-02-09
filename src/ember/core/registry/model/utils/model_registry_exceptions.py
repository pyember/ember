class ProviderError(Exception):
    """Base exception class for provider-related errors."""

    pass


class ProviderAPIError(ProviderError):
    """Exception raised when a provider API call fails."""

    pass


class ProviderConfigError(ProviderError):
    """Exception raised for errors in provider configuration."""

    pass


class InvalidPromptError(ProviderError):
    """Exception raised when the provided prompt is invalid."""

    pass
