"""Evaluation utilities for Ember core.

This package provides evaluators, pipelines, registries, and stateful evaluators for
assessing system outputs against expected values.
"""

from .base_evaluator import EvaluationResult, IEvaluator, IStatefulEvaluator
from .evaluators import (
    CodeExecutionEvaluator,
    ComposedEvaluator,
    ExactMatchEvaluator,
    NumericToleranceEvaluator,
    PartialRegexEvaluator,
)
from .pipeline import (
    BatchEvaluationSummary,
    PipelineEvaluator,
    evaluate_batch,
    evaluate_batch_with_summary,
)
from .registry import EvaluatorRegistry
from .stateful_evaluators import AggregatorEvaluator
