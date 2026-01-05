"""Base evaluator interfaces and result types.

This module defines the core protocol and datatypes used by the evaluation
subsystem, including a simple ``EvaluationResult``, a stateless ``IEvaluator``
interface, and a stateful ``IStatefulEvaluator`` interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

T_out = TypeVar("T_out")
T_ans = TypeVar("T_ans")


@dataclass
class EvaluationResult:
    """Result of evaluating system output.

    Attributes:
        is_correct: Whether output meets criteria.
        score: Numeric quality score.
        metadata: Additional evaluation details.
    """

    is_correct: bool
    score: float
    metadata: Mapping[str, object] | None = None


class IEvaluator(ABC, Generic[T_out, T_ans]):
    """Interface for output evaluation."""

    @abstractmethod
    def evaluate(
        self, system_output: T_out, correct_answer: T_ans, **kwargs: object
    ) -> EvaluationResult:
        """Evaluate system output against the expected answer.

        Args:
            system_output: Raw system output.
            correct_answer: Expected correct answer.
            **kwargs: Additional evaluator-specific options.

        Returns:
            Evaluation result with score and optional metadata.
        """
        raise NotImplementedError


class IStatefulEvaluator(ABC, Generic[T_out, T_ans]):
    """Evaluator that accumulates results across samples."""

    @abstractmethod
    def update(self, system_output: T_out, correct_answer: T_ans, **kwargs: object) -> None:
        """Add sample to internal state.

        Args:
            system_output: System output for the sample.
            correct_answer: Expected answer for the sample.
            **kwargs: Additional evaluator-specific options.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> EvaluationResult:
        """Compute the aggregated result from all samples.

        Returns:
            The aggregated evaluation result.
        """
        raise NotImplementedError

    def evaluate(
        self, system_output: T_out, correct_answer: T_ans, **kwargs: object
    ) -> EvaluationResult:
        """Evaluate a single sample by updating state and computing the result.

        Args:
            system_output: System output for the sample.
            correct_answer: Expected correct answer.
            **kwargs: Additional evaluator-specific options.

        Returns:
            The evaluation result for the sample.
        """
        self.update(system_output, correct_answer, **kwargs)
        return self.compute()
