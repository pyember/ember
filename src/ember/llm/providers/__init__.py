"""Provider implementations, one module per wire protocol.

`build` is the only entry point the client uses.
"""

from ember.context import Context, ProviderSettings
from ember.errors import ConfigError
from ember.llm.provider import Provider

__all__ = ["build"]


def build(settings: ProviderSettings, ctx: Context) -> Provider:
    """Constructs the provider described by ``settings``.

    The returned provider owns its HTTP client; the caller closes it.

    Raises:
        ConfigError: The provider cannot be constructed from ``settings``.
    """
    del ctx
    raise ConfigError(f"provider kind {settings.kind!r} is not implemented yet")
