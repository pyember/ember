from typing import Any, List

from .base_evaluator import EvaluationResult, IEvaluator, IStatefulEvaluator
from .pipeline import BatchEvaluationSummary, summarize_batch


class AggregatorEvaluator(IStatefulEvaluator[Any, Any]):
    """Evaluator that aggregates results from multiple samples.

    This evaluator wraps a stateless evaluator and accumulates individual evaluation results.
    It supports a two-phase evaluation process using the update and compute methods.

    Methods:
        update: Adds a new evaluation result.
        compute: Aggregates stored results and returns a final EvaluationResult.
        evaluate: Convenience method to update and compute for a single sample.
    """

    def __init__(self, evaluator: IEvaluator[Any, Any]) -> None:
        self.evaluator = evaluator
        self.results: List[EvaluationResult] = []

    def update(self, system_output: Any, correct_answer: Any, **kwargs: Any) -> None:
        """Accumulates the evaluation result for a given sample.

        Args:
            system_output (Any): The system-generated output.
            correct_answer (Any): The expected correct answer.
            **kwargs: Additional keyword arguments for evaluation.
        """
        result = self.evaluator.evaluate(system_output, correct_answer, **kwargs)
        self.results.append(result)

    def compute(self) -> EvaluationResult:
        """Aggregates all accumulated evaluation results into a final result.

        Returns:
            EvaluationResult: The aggregated evaluation result, including mean score and accuracy.
        """
        summary: BatchEvaluationSummary = summarize_batch(self.results)
        aggregated_correct = summary.accuracy == 1.0 if self.results else False
        return EvaluationResult(
            is_correct=aggregated_correct,
            score=summary.mean_score,
            metadata={"accuracy": summary.accuracy, "total_samples": len(self.results)},
        )

    def evaluate(
        self, system_output: Any, correct_answer: Any, **kwargs: Any
    ) -> EvaluationResult:
        """Updates the evaluator with a new sample and computes the aggregated result.

        Args:
            system_output (Any): The system output for the sample.
            correct_answer (Any): The expected answer for the sample.
            **kwargs: Additional keyword arguments for evaluation.

        Returns:
            EvaluationResult: The updated aggregated evaluation result.
        """
        self.update(system_output, correct_answer, **kwargs)
        return self.compute()
