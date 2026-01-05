"""Evaluation primitives and utilities."""

from .base_evaluator import EvaluationResult, IEvaluator, IStatefulEvaluator
from .evaluators import (
    CodeExecutionEvaluator,
    ComposedEvaluator,
    ExactMatchEvaluator,
    JsonValueEvaluator,
    MultipleChoiceStrictEvaluator,
    NumericToleranceEvaluator,
    PartialRegexEvaluator,
    PythonLiteralEqualityEvaluator,
    TokenListEvaluator,
    YesNoEvaluator,
)
from .pipeline import (
    BatchEvaluationSummary,
    PipelineEvaluator,
    evaluate_batch,
    evaluate_batch_with_summary,
)
from .registry import EvaluatorRegistry
from .stateful_evaluators import AggregatorEvaluator

__all__ = [
    "EvaluationResult",
    "IEvaluator",
    "IStatefulEvaluator",
    "CodeExecutionEvaluator",
    "ComposedEvaluator",
    "ExactMatchEvaluator",
    "JsonValueEvaluator",
    "MultipleChoiceStrictEvaluator",
    "NumericToleranceEvaluator",
    "PartialRegexEvaluator",
    "PythonLiteralEqualityEvaluator",
    "TokenListEvaluator",
    "YesNoEvaluator",
    "BatchEvaluationSummary",
    "PipelineEvaluator",
    "evaluate_batch",
    "evaluate_batch_with_summary",
    "EvaluatorRegistry",
    "AggregatorEvaluator",
]
