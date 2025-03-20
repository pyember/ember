"""Operator API for Ember (Singular Import Path)

This module re-exports from ember.api.operators to maintain backward compatibility with
code importing from ember.api.operator (singular).
"""

# Re-export all symbols from operators
from ember.api.operators import (  # Base classes; Type variables; Built-in operators; Useful types
    Any,
    Dict,
    EmberModel,
    EnsembleOperator,
    Field,
    InputT,
    JudgeSynthesisOperator,
    List,
    MostCommonAnswerSelector,
    Operator,
    Optional,
    OutputT,
    SelectorJudgeOperator,
    Specification,
    TypeVar,
    Union,
    VerifierOperator,
)

# Make everything available for import
__all__ = [
    # Base classes
    "Operator",
    "Specification",
    "EmberModel",
    "Field",
    # Type variables
    "InputT",
    "OutputT",
    # Built-in operators
    "EnsembleOperator",
    "MostCommonAnswerSelector",
    "VerifierOperator",
    "SelectorJudgeOperator",
    "JudgeSynthesisOperator",
    # Useful types
    "List",
    "Dict",
    "Any",
    "Optional",
    "Union",
    "TypeVar",
]
