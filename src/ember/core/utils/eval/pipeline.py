from __future__ import annotations
from typing import Any, Callable, List
from dataclasses import dataclass

from .base_evaluator import IEvaluator, EvaluationResult


class PipelineEvaluator(IEvaluator[Any, Any]):
    """Evaluator that applies a sequence of transformations before final evaluation.

    The evaluator applies each transformation function sequentially to the system output,
    then evaluates the final transformed value.

    Args:
        transforms (List[Callable[[Any], Any]]): A list of transformation functions.
        evaluator (IEvaluator[Any, Any]): The evaluator to be applied on the transformed output.
    """

    def __init__(
        self, transforms: List[Callable[[Any], Any]], evaluator: IEvaluator[Any, Any]
    ) -> None:
        self.transforms = transforms
        self.evaluator = evaluator

    def evaluate(
        self, system_output: Any, correct_answer: Any, **kwargs: Any
    ) -> EvaluationResult:
        """Evaluates the system output after applying a sequence of transformations.

        Args:
            system_output (Any): The initial system output.
            correct_answer (Any): The expected result after transformation.
            **kwargs: Additional keyword arguments.

        Returns:
            EvaluationResult: The evaluation result from the final evaluator.
        """
        transformed_value = system_output
        for transform in self.transforms:
            transformed_value = transform(transformed_value)
        return self.evaluator.evaluate(transformed_value, correct_answer, **kwargs)


@dataclass
class BatchEvaluationSummary:
    """Aggregated summary of a batch evaluation.

    Attributes:
        results (List[EvaluationResult]): List of individual evaluation results.
        mean_score (float): Average score computed across all evaluations.
        accuracy (float): Proportion of evaluations that were correct.
    """

    results: List[EvaluationResult]
    mean_score: float
    accuracy: float


def summarize_batch(results: List[EvaluationResult]) -> BatchEvaluationSummary:
    """Computes the average score and accuracy from a list of EvaluationResults.

    Args:
        results (List[EvaluationResult]): A list of evaluation results.

    Returns:
        BatchEvaluationSummary: An aggregated summary including mean score and accuracy.
    """
    total_score = sum(r.score for r in results)
    count = len(results)
    mean_score = total_score / count if count else 0.0
    accuracy = sum(1 for r in results if r.is_correct) / count if count else 0.0
    return BatchEvaluationSummary(
        results=results, mean_score=mean_score, accuracy=accuracy
    )


def evaluate_batch(
    evaluator: IEvaluator[Any, Any],
    system_outputs: List[Any],
    correct_answers: List[Any],
    **kwargs: Any,
) -> List[EvaluationResult]:
    """Evaluates a batch of system outputs against their corresponding correct answers.

    Args:
        evaluator (IEvaluator[Any, Any]): The evaluator to apply.
        system_outputs (List[Any]): List of system outputs.
        correct_answers (List[Any]): List of corresponding correct answers.
        **kwargs: Additional keyword arguments for evaluation.

    Returns:
        List[EvaluationResult]: A list of individual evaluation results.

    Raises:
        ValueError: If system_outputs and correct_answers have different lengths.
    """
    if len(system_outputs) != len(correct_answers):
        raise ValueError(
            f"Mismatched list lengths: system_outputs ({len(system_outputs)}) and "
            f"correct_answers ({len(correct_answers)}) must have the same length."
        )

    results = []
    for output, answer in zip(system_outputs, correct_answers):
        results.append(evaluator.evaluate(output, answer, **kwargs))
    return results


def evaluate_batch_with_summary(
    evaluator: IEvaluator[Any, Any],
    system_outputs: List[Any],
    correct_answers: List[Any],
    **kwargs: Any,
) -> BatchEvaluationSummary:
    """Evaluates a batch of samples and returns an aggregated summary.

    Args:
        evaluator (IEvaluator[Any, Any]): The evaluator to apply.
        system_outputs (List[Any]): List of system outputs.
        correct_answers (List[Any]): List of expected answers.
        **kwargs: Additional keyword arguments for evaluation.

    Returns:
        BatchEvaluationSummary: Aggregated summary containing mean score and accuracy.
    """
    results = evaluate_batch(
        evaluator=evaluator,
        system_outputs=system_outputs,
        correct_answers=correct_answers,
        **kwargs,
    )
    return summarize_batch(results)
