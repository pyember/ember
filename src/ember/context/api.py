"""Public helpers for retrieving or overriding the active Ember context.

The helpers wrap :mod:`ember._internal.context` to provide a minimal, opinionated
API for application and test code.

Examples:
    >>> from ember.context import context
    >>> context.get().get_config("models.default")
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from ember._internal.context import EmberContext as _EmberContext
from ember._internal.context import current_context as _current_context

EmberContext = _EmberContext


class ContextAPI:
    """Expose thread- and task-safe context operations through one namespace."""

    @staticmethod
    def get() -> _EmberContext:
        """Return the currently active context.

        Returns:
            EmberContext: Context for the running task or request.

        Examples:
            >>> from ember.context import context
            >>> context.get().get_config("models.default")
        """

        return _current_context()

    @staticmethod
    @contextmanager
    def manager(**config_overrides: Any):
        """Yield a child context with temporary configuration overrides.

        Args:
            **config_overrides: Keyword overrides applied to the child context.

        Yields:
            EmberContext: Context carrying the requested overrides.

        Examples:
            >>> from ember.context import context
            >>> with context.manager(models={"default": "gpt-4"}) as ctx:
            ...     ctx.get_config("models.default")
        """

        current = _current_context()
        child = current.create_child(**config_overrides)
        with child as ctx:
            yield ctx


context = ContextAPI()


def get_config(key: str, default: Any | None = None) -> Any | None:
    """Look up a configuration value from the active context.

    Args:
        key: Dot-delimited configuration path such as ``"models.default"``.
        default: Value returned when ``key`` is missing.

    Returns:
        Any | None: Configured value or ``default`` when the key is absent.

    Examples:
        >>> from ember.context import get_config
        >>> get_config("models.default")
    """

    return context.get().get_config(key, default)


def set_config(key: str, value: Any) -> None:
    """Persist a configuration value on the active context.

    Args:
        key: Dot-delimited configuration path such as ``"models.temperature"``.
        value: Value stored on the current context.

    Examples:
        >>> from ember.context import set_config
        >>> set_config("models.temperature", 0.7)
    """

    context.get().set_config(key, value)


__all__ = [
    "ContextAPI",
    "EmberContext",
    "context",
    "get_config",
    "set_config",
]
