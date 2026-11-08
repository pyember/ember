"""Errors raised by ember.

Ember raises a small, closed set of exceptions. A failed model call is always a
`ModelError` whose `kind` says what went wrong; callers branch on `kind` and
`retryable`, never on message text.
"""

import enum

__all__ = [
    "ConfigError",
    "EmberError",
    "ErrorKind",
    "ModelError",
    "NonSyntaxError",
]


class EmberError(Exception):
    """Base class for every exception ember raises on purpose."""


class ConfigError(EmberError):
    """Configuration is unreadable or invalid, or a required credential is missing."""


class ErrorKind(enum.Enum):
    """Why a model call failed."""

    AUTH = "auth"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    BAD_REQUEST = "bad_request"
    RATE_LIMITED = "rate_limited"
    OVERLOADED = "overloaded"
    SERVER = "server"
    TRANSPORT = "transport"
    UNPRICED = "unpriced"
    CANCELLED = "cancelled"


_RETRYABLE_KINDS = frozenset(
    {ErrorKind.RATE_LIMITED, ErrorKind.OVERLOADED, ErrorKind.SERVER, ErrorKind.TRANSPORT}
)


class ModelError(EmberError):
    """A model call failed.

    The message is built only from the fields below and a short reason written
    by ember. It never contains upstream response bodies, file paths, or
    credentials.

    Attributes:
        kind: Why the call failed.
        provider: Provider name, for example ``"openai"``.
        model: Model name as the caller wrote it.
        reason: Short human-readable explanation written by ember.
        status: HTTP status code, when the failure came from an HTTP response.
        request_id: The provider's request id, when the provider sent one.
        retry_after_s: Seconds the provider asked the caller to wait.
        before_first_byte: For transport failures, whether no response byte had
            arrived yet. Only such failures are safe to retry.
    """

    def __init__(
        self,
        kind: ErrorKind,
        *,
        provider: str,
        model: str,
        reason: str,
        status: int | None = None,
        request_id: str | None = None,
        retry_after_s: float | None = None,
        before_first_byte: bool = True,
    ) -> None:
        self.kind = kind
        self.provider = provider
        self.model = model
        self.reason = reason
        self.status = status
        self.request_id = request_id
        self.retry_after_s = retry_after_s
        self.before_first_byte = before_first_byte
        super().__init__(self._render())

    @property
    def retryable(self) -> bool:
        """Whether sending the same request again can succeed."""
        if self.kind is ErrorKind.TRANSPORT:
            return self.before_first_byte
        return self.kind in _RETRYABLE_KINDS

    def _render(self) -> str:
        parts = [f"{self.kind.value}: {self.reason}", f"model={self.provider}/{self.model}"]
        if self.status is not None:
            parts.append(f"status={self.status}")
        if self.request_id is not None:
            parts.append(f"request_id={self.request_id}")
        if self.retry_after_s is not None:
            parts.append(f"retry_after_s={self.retry_after_s:g}")
        return " ".join(parts)


class NonSyntaxError(EmberError, ValueError):
    """A NON graph specification is invalid.

    Attributes:
        spec: The specification that failed to parse.
        column: Zero-based column where the error was detected.
        reason: What is wrong at that column.
    """

    def __init__(self, spec: str, column: int, reason: str) -> None:
        self.spec = spec
        self.column = column
        self.reason = reason
        pointer = " " * column + "^"
        super().__init__(f"{reason} at column {column}\n  {spec}\n  {pointer}")
