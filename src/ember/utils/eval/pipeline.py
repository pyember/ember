"""Evaluation pipelines and batch helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from .base_evaluator import EvaluationResult, IEvaluator

T_out = TypeVar("T_out")
T_ans = TypeVar("T_ans")


class PipelineEvaluator(IEvaluator[T_out, T_ans]):
    """Apply transformations before final evaluation.

    Each transformation function is applied sequentially to the system output,
    then the final transformed value is evaluated.

    Args:
        transforms: Transformation functions to apply in order.
        evaluator: Evaluator applied to the transformed output.
    """

    def __init__(
        self,
        transforms: Sequence[Callable[[T_out], T_out]],
        evaluator: IEvaluator[T_out, T_ans],
    ) -> None:
        self.transforms = list(transforms)
        self.evaluator = evaluator

    def evaluate(
        self, system_output: T_out, correct_answer: T_ans, **kwargs: object
    ) -> EvaluationResult:
        """Evaluate the system output after applying transformations.

        Args:
            system_output: Initial system output.
            correct_answer: Expected result after transformation.
            **kwargs: Additional evaluator-specific options.

        Returns:
            The evaluation result from the final evaluator.
        """
        transformed_value = system_output
        for transform in self.transforms:
            transformed_value = transform(transformed_value)
        return self.evaluator.evaluate(transformed_value, correct_answer, **kwargs)


@dataclass
class BatchEvaluationSummary:
    """Aggregated summary of a batch evaluation.

    Attributes:
        results: Individual evaluation results.
        mean_score: Average score across all evaluations.
        accuracy: Proportion of evaluations that were correct.
    """

    results: list[EvaluationResult]
    mean_score: float
    accuracy: float


def summarize_batch(results: Sequence[EvaluationResult]) -> BatchEvaluationSummary:
    """Compute average score and accuracy for a batch.

    Args:
        results: Evaluation results.

    Returns:
        Aggregated summary including mean score and accuracy.
    """
    total_score = sum(r.score for r in results)
    count = len(results)
    mean_score = total_score / count if count else 0.0
    accuracy = sum(1 for r in results if r.is_correct) / count if count else 0.0
    return BatchEvaluationSummary(results=list(results), mean_score=mean_score, accuracy=accuracy)


def evaluate_batch(
    evaluator: IEvaluator[T_out, T_ans],
    system_outputs: Sequence[T_out],
    correct_answers: Sequence[T_ans],
    **kwargs: object,
) -> list[EvaluationResult]:
    """Evaluate a batch against corresponding correct answers.

    Args:
        evaluator: Evaluator to apply.
        system_outputs: System outputs.
        correct_answers: Corresponding correct answers.
        **kwargs: Additional keyword arguments for evaluation.

    Returns:
        Individual evaluation results.

    Raises:
        ValueError: If the lists have different lengths.
    """
    if len(system_outputs) != len(correct_answers):
        raise ValueError(
            f"Mismatched list lengths: system_outputs ({len(system_outputs)}) and "
            f"correct_answers ({len(correct_answers)}) must have the same length."
        )

    return [
        evaluator.evaluate(output, answer, **kwargs)
        for output, answer in zip(system_outputs, correct_answers, strict=True)
    ]


def evaluate_batch_with_summary(
    evaluator: IEvaluator[T_out, T_ans],
    system_outputs: Sequence[T_out],
    correct_answers: Sequence[T_ans],
    **kwargs: object,
) -> BatchEvaluationSummary:
    """Evaluate a batch of samples and return an aggregated summary.

    Args:
        evaluator: Evaluator to apply.
        system_outputs: System outputs.
        correct_answers: Expected answers.
        **kwargs: Additional keyword arguments for evaluation.

    Returns:
        Aggregated summary containing mean score and accuracy.
    """
    results = evaluate_batch(
        evaluator=evaluator,
        system_outputs=system_outputs,
        correct_answers=correct_answers,
        **kwargs,
    )
    return summarize_batch(results)
