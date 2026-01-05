"""Ember model runtime primitives and typed schemas.

This package consolidates the registry, pricing helpers, and shared dataclasses
used across Ember's model stack. Provider SDKs remain lazily imported by the
provider registry; the symbols re-exported here are explicit (no module-level
dynamic attribute dispatch).
"""

from __future__ import annotations

from ember._internal.concurrency import (
    AdaptiveConcurrencyLimiter,
    ConcurrencyKey,
    Outcome,
)
from ember._internal.concurrency.limiter import get_limiter, outcome_from_error
from ember.models.pricing import get_model_cost, get_model_costs
from ember.models.runtime.registry import ModelRegistry
from ember.models.schemas import (
    ChatRequest,
    ChatResponse,
    EmbeddingResponse,
    ModelCost,
    ModelInfo,
    ProviderInfo,
    ProviderParams,
    RateLimit,
    UsageRecord,
    UsageStats,
    UsageSummary,
)
from ember.models.tokenization import count_tokens

__all__ = [
    # Core components
    "ModelRegistry",
    # Adaptive concurrency control
    "AdaptiveConcurrencyLimiter",
    "ConcurrencyKey",
    "Outcome",
    "get_limiter",
    "outcome_from_error",
    # Cost functions
    "get_model_costs",
    "count_tokens",
    "get_model_cost",
    # Schemas - Provider
    "ProviderInfo",
    "ProviderParams",
    # Schemas - Model
    "ModelInfo",
    "ModelCost",
    "RateLimit",
    # Schemas - Request/Response
    "ChatRequest",
    "ChatResponse",
    "EmbeddingResponse",
    # Schemas - Usage
    "UsageStats",
    "UsageRecord",
    "UsageSummary",
]
