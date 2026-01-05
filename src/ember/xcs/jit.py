"""Compatibility shim re-exporting XCS JIT entrypoints."""

from ember.xcs.api import get_jit_stats, jit

__all__ = ("jit", "get_jit_stats")
