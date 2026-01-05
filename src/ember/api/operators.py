"""Public entry points for constructing and composing Ember operators.

The module re-exports the most common operator types plus the ``@op`` decorator so
callers can build pipelines without importing from internal packages.

Examples:
    >>> from ember.api import operators
    >>>
    >>> @operators.op
    ... def preprocess(text: str) -> str:
    ...     return text.strip()
    >>>
    >>> pipeline = operators.chain(preprocess, operators.ModelText("gpt-4"))
"""

from ember.api.decorators import op
from ember.operators import (
    Chain,
    Ensemble,
    ExtractText,
    LearnableRouter,
    ModelCall,
    ModelText,
    Operator,
    Router,
    chain,
    ensemble,
    router,
)

__all__ = [
    "Operator",
    "op",
    # Common operators
    "ModelCall",
    "Ensemble",
    "Chain",
    "Router",
    "LearnableRouter",
    "ModelText",
    "ExtractText",
    # Convenience functions
    "chain",
    "ensemble",
    "router",
]
