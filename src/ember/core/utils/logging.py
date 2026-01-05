"""Logging utilities for Ember (thin wrappers).

This module provides a stable import path for internal code while delegating
actual behavior to ``ember.utils.logging`` to avoid configuration drift.
"""

from __future__ import annotations

import logging
from typing import Union

from ember.utils.logging import configure_logging as _configure_logging
from ember.utils.logging import set_component_level as _set_component_level


def get_logger(name: str) -> logging.Logger:
    """Get a module logger by name (no prefixing).

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


def configure_logging(verbose: bool = False) -> None:
    """Configure logging via the central logger configuration.

    Args:
        verbose: Enable more detailed logging when True.
    """
    _configure_logging(verbose=verbose)


def set_component_level(component: str, level: Union[str, int]) -> None:
    """Set log level for a specific component or group.

    Accepts either string levels (e.g., "INFO") or numeric constants.
    """
    if isinstance(level, str):
        level_value = getattr(logging, level.upper(), logging.INFO)
    else:
        level_value = level
    _set_component_level(component, level_value)
