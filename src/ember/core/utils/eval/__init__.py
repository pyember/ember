"""Evaluation utilities for Ember core.

This package provides evaluators, pipelines, registries, and stateful evaluators for
assessing system outputs against expected values.
"""

from .base_evaluator import IEvaluator, IStatefulEvaluator, EvaluationResult
from .evaluators import (
    ExactMatchEvaluator,
    NumericToleranceEvaluator,
    CodeExecutionEvaluator,
    ComposedEvaluator,
    PartialRegexEvaluator,
)
from .pipeline import (
    PipelineEvaluator,
    BatchEvaluationSummary,
    evaluate_batch,
    evaluate_batch_with_summary,
)
from .registry import EvaluatorRegistry
from .stateful_evaluators import AggregatorEvaluator
