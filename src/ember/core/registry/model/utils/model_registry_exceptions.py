class ProviderError(Exception):
    """Generic parent exception for errors raised by LLM providers."""


class ProviderAPIError(ProviderError):
    """Raised when the provider's API call fails (network, invalid request, etc.)."""


class ProviderConfigError(ProviderError):
    """Raised when the provider configuration (API key, settings) is invalid."""


class InvalidPromptError(ProviderError):
    """Raised when the user prompt is empty or otherwise invalid."""
