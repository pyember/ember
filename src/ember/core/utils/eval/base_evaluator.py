from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar

# We define separate type variables for system output and correct answer.
T_out = TypeVar("T_out")
T_ans = TypeVar("T_ans")


@dataclass
class EvaluationResult:
    """Encapsulates the result of evaluating a system output against a reference.

    Attributes:
        is_correct (bool): Whether the system output meets the expected criteria.
        score (float): Numeric score reflecting accuracy or quality.
        metadata (Optional[Dict[str, Any]]): Additional details about the evaluation.
    """

    is_correct: bool
    score: float
    metadata: Optional[Dict[str, Any]] = None


class IEvaluator(ABC, Generic[T_out, T_ans]):
    """Interface for evaluating a system output against a correct answer.

    Subclasses should override the evaluate method.

    Methods:
        evaluate: Compares system output with the expected answer and returns an EvaluationResult.
    """

    @abstractmethod
    def evaluate(
        self, system_output: T_out, correct_answer: T_ans, **kwargs: Any
    ) -> EvaluationResult:
        """Evaluates the system output against the expected correct answer.

        Args:
            system_output (T_out): The raw output from the system.
            correct_answer (T_ans): The expected correct answer.
            **kwargs: Additional keyword arguments.

        Returns:
            EvaluationResult: The evaluation result.
        """
        raise NotImplementedError


class IStatefulEvaluator(ABC, Generic[T_out, T_ans]):
    """Interface for evaluators that accumulate results across multiple samples.

    Implements a two-phase evaluation: first updating internal state and then computing the aggregated result.

    Methods:
        update: Accumulates state with a new sample.
        compute: Computes and returns the final aggregated EvaluationResult.
    """

    @abstractmethod
    def update(
        self, system_output: T_out, correct_answer: T_ans, **kwargs: Any
    ) -> None:
        """Accumulates internal state with a new sample evaluation.

        Args:
            system_output (T_out): The system output for the sample.
            correct_answer (T_ans): The expected correct answer for the sample.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> EvaluationResult:
        """Computes the aggregated evaluation result from all accumulated samples.

        Returns:
            EvaluationResult: The aggregated evaluation result.
        """
        raise NotImplementedError

    def evaluate(
        self, system_output: T_out, correct_answer: T_ans, **kwargs: Any
    ) -> EvaluationResult:
        """Convenience method for single-sample evaluation: updates state and computes the result.

        Args:
            system_output (T_out): The system output for the sample.
            correct_answer (T_ans): The expected correct answer.
            **kwargs: Additional keyword arguments.

        Returns:
            EvaluationResult: The evaluation result for the sample.
        """
        self.update(system_output, correct_answer, **kwargs)
        return self.compute()
