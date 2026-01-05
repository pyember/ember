"""Lightweight progress reporting utilities.

This module provides a minimal, dependency-free progress reporter API so callers
can toggle quiet/verbose behavior without pulling in heavy UI libraries.

Current usage in the codebase only toggles quiet mode and does not rely on the
return value. A simple reporter object is returned for future extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProgressReporter:
    """A no-op progress reporter with a familiar interface.

    Methods are intentionally no-ops so the reporter can be safely used in any
    environment (CI, TTY, non-TTY) without side effects.
    """

    quiet: bool = False

    def start(self, message: Optional[str] = None) -> None:  # pragma: no cover - trivial
        if not self.quiet and message:
            print(message)

    def update(self, message: str) -> None:  # pragma: no cover - trivial
        if not self.quiet:
            print(message)

    def done(self, message: Optional[str] = None) -> None:  # pragma: no cover - trivial
        if not self.quiet and message:
            print(message)


def get_default_reporter(quiet: bool = False) -> ProgressReporter:
    """Return a basic progress reporter.

    Args:
        quiet: When True, suppresses all progress output.

    Returns:
        A no-op reporter honoring the given quiet flag.
    """
    return ProgressReporter(quiet=quiet)
