"""Re-export the public context helpers used by Ember applications.

Examples:
    >>> from ember.context import context
    >>> context.get()

    >>> from ember.context import context
    >>> with context.manager(models={"default": "gpt-4"}):
    ...     ...
"""

from .api import ContextAPI, EmberContext, context, get_config, set_config

__all__ = [
    "ContextAPI",
    "EmberContext",
    "context",
    "get_config",
    "set_config",
]
