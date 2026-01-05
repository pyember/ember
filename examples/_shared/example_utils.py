"""Shared utilities for Ember examples."""

from __future__ import annotations


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    width = max(50, len(title) + 4)
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")
