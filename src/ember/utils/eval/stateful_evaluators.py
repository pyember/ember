"""Stateful evaluators that aggregate results across samples."""

from __future__ import annotations

from typing import TypeVar

from .base_evaluator import EvaluationResult, IEvaluator, IStatefulEvaluator
from .pipeline import BatchEvaluationSummary, summarize_batch

T_out = TypeVar("T_out")
T_ans = TypeVar("T_ans")


class AggregatorEvaluator(IStatefulEvaluator[T_out, T_ans]):
    """Aggregates evaluation results across samples.

    Two-phase evaluation: accumulate with update(), finalize with compute().
    """

    def __init__(self, evaluator: IEvaluator[T_out, T_ans]) -> None:
        self.evaluator = evaluator
        self.results: list[EvaluationResult] = []

    def update(self, system_output: T_out, correct_answer: T_ans, **kwargs: object) -> None:
        """Add evaluation result for a sample.

        Args:
            system_output: Model output.
            correct_answer: Expected answer.
            **kwargs: Additional evaluator-specific options.
        """
        result = self.evaluator.evaluate(system_output, correct_answer, **kwargs)
        self.results.append(result)

    def compute(self) -> EvaluationResult:
        """Aggregate all results into final score.

        Returns:
            Combined evaluation with mean score and accuracy.
        """
        summary: BatchEvaluationSummary = summarize_batch(self.results)
        aggregated_correct = summary.accuracy == 1.0 if self.results else False
        return EvaluationResult(
            is_correct=aggregated_correct,
            score=summary.mean_score,
            metadata={"accuracy": summary.accuracy, "total_samples": len(self.results)},
        )
