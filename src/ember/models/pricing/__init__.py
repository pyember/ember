"""Public pricing API surface for ``ember.models.pricing``.

Examples:
    >>> from ember.models.pricing import get_model_cost
    >>> sorted(get_model_cost("unknown", strict=False))
    ['context', 'input', 'output']

CLI helpers for refreshing ``pricing.yaml`` live in :mod:`ember.models.pricing_updater`.
"""

from __future__ import annotations

from .manager import (
    Pricing,
    _pricing,
    get_model_cost,
    get_model_costs,
    get_model_pricing,
)
from .tracker import (
    CostTracker,
    get_cost_accuracy,
    track_usage,
)

__all__ = [
    "Pricing",
    "get_model_cost",
    "get_model_costs",
    "get_model_pricing",
    "_pricing",
    "CostTracker",
    "track_usage",
    "get_cost_accuracy",
]
