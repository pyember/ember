"""Backward-compatible import surface for XCS analysis helpers."""

from __future__ import annotations

from ember.xcs.compiler.analysis import (
    OperationSummary,
    analyze_operations,
    has_jax_arrays,
    is_jax_array,
)

OperationType = OperationSummary

__all__ = [
    "OperationSummary",
    "OperationType",
    "analyze_operations",
    "has_jax_arrays",
    "is_jax_array",
]
