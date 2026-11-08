"""The active ember configuration.

Configuration lives in one immutable `Context` held by one contextvar. The
first call to `current()` builds the root context from the config file; that
is the first moment ember reads disk. `override()` layers changes for the
duration of a ``with`` block and never writes them anywhere.

Example:
    >>> from ember import context
    >>> with context.override(require_price=True):
    ...     context.current().require_price
    True
"""

import contextlib
import contextvars
import dataclasses
import os
import threading
from collections.abc import Generator, Mapping
from types import MappingProxyType
from typing import Literal

from ember import config as config_file
from ember.errors import ConfigError

__all__ = [
    "BUILTIN_PROVIDERS",
    "Context",
    "ProviderSettings",
    "RetryPolicy",
    "Secret",
    "current",
    "override",
    "reload",
]

ProviderKind = Literal["openai", "anthropic", "gemini", "openai_compatible"]

#: Providers available without any configuration, with the environment
#: variable each one reads its key from.
BUILTIN_PROVIDERS: Mapping[str, tuple[ProviderKind, str]] = MappingProxyType(
    {
        "openai": ("openai", "OPENAI_API_KEY"),
        "anthropic": ("anthropic", "ANTHROPIC_API_KEY"),
        "gemini": ("gemini", "GEMINI_API_KEY"),
    }
)


class Secret:
    """A credential that never appears in reprs, logs or error messages."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not value:
            raise ConfigError("a credential cannot be empty")
        self._value = value

    def reveal(self) -> str:
        """Returns the credential. Call only where it is sent to its provider."""
        return self._value

    def __repr__(self) -> str:
        return "Secret(…)" if len(self._value) < 12 else f"Secret('…{self._value[-4:]}')"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Secret) and other._value == self._value

    def __hash__(self) -> int:
        return hash(self._value)


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderSettings:
    """How to reach one provider.

    Attributes:
        name: The provider name used in model ids, for example ``"openai"`` in
            ``"openai/gpt-5"``.
        kind: Which wire protocol the provider speaks.
        api_key_env: Environment variable holding the key, if any.
        api_key: A key given in the config file or an override.
        api_key_from_override: Whether `api_key` came from `override()`, which
            takes precedence over the environment.
        base_url: Endpoint root; ``None`` uses the provider's public endpoint.
    """

    name: str
    kind: ProviderKind
    api_key_env: str | None = None
    api_key: Secret | None = None
    api_key_from_override: bool = False
    base_url: str | None = None

    def credential(self) -> Secret | None:
        """Resolves the key: override, then environment, then config file.

        Returns:
            The key, or ``None`` when the provider has no key configured
            anywhere. Callers decide whether a key is required.
        """
        if self.api_key is not None and self.api_key_from_override:
            return self.api_key
        if self.api_key_env:
            value = os.environ.get(self.api_key_env)
            if value:
                return Secret(value)
        return self.api_key


@dataclasses.dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How the client retries retryable model errors.

    Attributes:
        max_attempts: Total attempts per request, including the first.
        base_delay_s: Delay before the first retry; doubles per attempt.
        max_delay_s: Upper bound on any single delay, applied after jitter.
    """

    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 20.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ConfigError("retry.max_attempts must be at least 1")
        if not 0 <= self.base_delay_s <= self.max_delay_s:
            raise ConfigError("retry delays must satisfy 0 <= base_delay_s <= max_delay_s")


@dataclasses.dataclass(frozen=True, slots=True)
class Context:
    """Everything a model call needs to know about its environment.

    Attributes:
        providers: Provider settings by name.
        default_model: Model used when a call names none.
        require_price: Refuse to send a request whose price is unknown.
        max_in_flight_per_provider: Concurrent requests allowed per provider.
        retry: Retry policy for retryable errors.
        connect_timeout_s: Bound on establishing a connection.
        idle_timeout_s: Bound on waiting for the next response byte. This is a
            liveness bound, not a cap on total generation time.
    """

    providers: Mapping[str, ProviderSettings]
    default_model: str | None = None
    require_price: bool = False
    max_in_flight_per_provider: int = 64
    retry: RetryPolicy = RetryPolicy()
    connect_timeout_s: float = 10.0
    idle_timeout_s: float = 300.0

    def provider(self, name: str) -> ProviderSettings:
        """Returns the settings for ``name``.

        Raises:
            ConfigError: No provider with that name is configured.
        """
        try:
            return self.providers[name]
        except KeyError:
            known = ", ".join(sorted(self.providers))
            raise ConfigError(f"unknown provider {name!r}; configured: {known}") from None


_ACTIVE: contextvars.ContextVar[Context] = contextvars.ContextVar("ember_context")
_root_lock = threading.Lock()
_root: Context | None = None


def current() -> Context:
    """Returns the active context, building the root from the config file once.

    Raises:
        ConfigError: The config file exists but cannot be read or is invalid.
    """
    active = _ACTIVE.get(None)
    if active is not None:
        return active
    return _root_context()


def reload() -> None:
    """Forgets the root context so the next `current()` re-reads the config file."""
    global _root
    with _root_lock:
        _root = None


@contextlib.contextmanager
def override(
    *,
    providers: Mapping[str, Mapping[str, str]] | None = None,
    default_model: str | None = None,
    require_price: bool | None = None,
    max_in_flight_per_provider: int | None = None,
    retry: RetryPolicy | None = None,
) -> Generator[Context, None, None]:
    """Applies changes to the active context for the duration of a block.

    Provider overrides are merged field by field, so overriding ``api_key``
    keeps the configured ``base_url``. Overrides are never written to disk.

    Args:
        providers: Per-provider fields to change: ``api_key``, ``base_url``,
            ``api_key_env`` and, for new providers, ``kind``.
        default_model: Replaces the default model.
        require_price: Replaces the price policy.
        max_in_flight_per_provider: Replaces the concurrency limit.
        retry: Replaces the retry policy.

    Yields:
        The new active context.
    """
    base = current()
    layered = dataclasses.replace(
        base,
        providers=(
            base.providers
            if providers is None
            else _merge_providers(base.providers, providers, from_override=True)
        ),
        default_model=base.default_model if default_model is None else default_model,
        require_price=base.require_price if require_price is None else require_price,
        max_in_flight_per_provider=(
            base.max_in_flight_per_provider
            if max_in_flight_per_provider is None
            else _positive("max_in_flight_per_provider", max_in_flight_per_provider)
        ),
        retry=base.retry if retry is None else retry,
    )
    token = _ACTIVE.set(layered)
    try:
        yield layered
    finally:
        _ACTIVE.reset(token)


def _root_context() -> Context:
    global _root
    with _root_lock:
        if _root is None:
            _root = _context_from_file(config_file.load())
        return _root


def _context_from_file(data: config_file.ConfigData) -> Context:
    builtin = {
        name: ProviderSettings(name=name, kind=kind, api_key_env=env)
        for name, (kind, env) in BUILTIN_PROVIDERS.items()
    }
    providers = _merge_providers(builtin, data.providers, from_override=False)
    return Context(
        providers=providers,
        default_model=data.default_model,
        require_price=data.require_price,
        max_in_flight_per_provider=_positive(
            "max_in_flight_per_provider", data.max_in_flight_per_provider
        ),
        retry=_retry_policy(data.retry),
    )


def _merge_providers(
    base: Mapping[str, ProviderSettings],
    changes: Mapping[str, Mapping[str, str]],
    *,
    from_override: bool,
) -> Mapping[str, ProviderSettings]:
    merged = dict(base)
    for name, fields in changes.items():
        unknown = set(fields) - {"kind", "api_key", "api_key_env", "base_url"}
        if unknown:
            raise ConfigError(f"provider {name!r}: unknown fields {sorted(unknown)}")
        existing = merged.get(name)
        if existing is None:
            existing = ProviderSettings(name=name, kind=_provider_kind(name, fields.get("kind")))
        elif "kind" in fields and fields["kind"] != existing.kind:
            raise ConfigError(f"provider {name!r} is {existing.kind}; its kind cannot change")
        has_key = "api_key" in fields
        merged[name] = ProviderSettings(
            name=name,
            kind=existing.kind,
            api_key_env=fields.get("api_key_env", existing.api_key_env),
            api_key=Secret(fields["api_key"]) if has_key else existing.api_key,
            api_key_from_override=from_override if has_key else existing.api_key_from_override,
            base_url=fields.get("base_url", existing.base_url),
        )
    return MappingProxyType(merged)


_KINDS: Mapping[str, ProviderKind] = MappingProxyType(
    {
        "openai": "openai",
        "anthropic": "anthropic",
        "gemini": "gemini",
        "openai_compatible": "openai_compatible",
    }
)


def _provider_kind(name: str, kind: str | None) -> ProviderKind:
    if kind is None or kind not in _KINDS:
        raise ConfigError(f"provider {name!r} needs kind: {', '.join(_KINDS)}")
    return _KINDS[kind]


def _retry_policy(raw: Mapping[str, float]) -> RetryPolicy:
    default = RetryPolicy()
    attempts = raw.get("max_attempts", default.max_attempts)
    if attempts != int(attempts):
        raise ConfigError(f"retry.max_attempts must be an integer, got {attempts}")
    return RetryPolicy(
        max_attempts=int(attempts),
        base_delay_s=float(raw.get("base_delay_s", default.base_delay_s)),
        max_delay_s=float(raw.get("max_delay_s", default.max_delay_s)),
    )


def _positive(name: str, value: int) -> int:
    if value < 1:
        raise ConfigError(f"{name} must be at least 1, got {value}")
    return value
