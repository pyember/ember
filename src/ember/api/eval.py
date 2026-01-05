"""Evaluation helpers and lightweight evaluator registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, TypeAlias

from ember.utils.eval.base_evaluator import EvaluationResult, IEvaluator
from ember.utils.eval.registry import EvaluatorRegistry

_REGISTRY = EvaluatorRegistry()
_INITIALIZED = False


def _ensure_registry_initialized() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    from ember.utils.eval.code_execution import CodeCompetitionEvaluator
    from ember.utils.eval.evaluators import (
        BBEHEvaluator,
        ExactMatchEvaluator,
        JsonValueEvaluator,
        MultipleChoiceStrictEvaluator,
        NumericToleranceEvaluator,
        PartialRegexEvaluator,
        PythonLiteralEqualityEvaluator,
        TokenListEvaluator,
        YesNoEvaluator,
    )
    from ember.utils.eval.numeric_answer import AIMEAnswerEvaluator, NumericAnswerEvaluator

    _REGISTRY.register("exact_match", ExactMatchEvaluator)
    _REGISTRY.register("accuracy", ExactMatchEvaluator)  # Alias for exact_match
    _REGISTRY.register("bbeh", BBEHEvaluator)
    _REGISTRY.register("numeric", NumericToleranceEvaluator)
    _REGISTRY.register("numeric_answer", NumericAnswerEvaluator)
    _REGISTRY.register("aime", AIMEAnswerEvaluator)
    _REGISTRY.register("regex", PartialRegexEvaluator)
    _REGISTRY.register("python_literal", PythonLiteralEqualityEvaluator)
    _REGISTRY.register("multiple_choice_strict", MultipleChoiceStrictEvaluator)
    _REGISTRY.register("yes_no", YesNoEvaluator)
    _REGISTRY.register("token_list", TokenListEvaluator)
    _REGISTRY.register("json_value", JsonValueEvaluator)
    _REGISTRY.register("code_competition", CodeCompetitionEvaluator)

    _INITIALIZED = True


def list_available_evaluators() -> list[str]:
    """Return the sorted evaluator names currently registered."""
    _ensure_registry_initialized()
    return sorted(_REGISTRY.names())


def register_evaluator(name: str, evaluator_factory: Callable[..., IEvaluator]) -> None:
    """Register an evaluator factory for lookup via :meth:`Evaluator.from_registry`."""
    _ensure_registry_initialized()
    _REGISTRY.register(name, evaluator_factory)


NumericMetric: TypeAlias = int | float


class MetricFunction(Protocol):
    def __call__(
        self, prediction: object, reference: object, **kwargs: object
    ) -> dict[str, object]: ...


class Evaluator:
    """Thin wrapper that normalizes evaluator outputs into dictionaries."""

    @classmethod
    def from_registry(cls, name: str, **kwargs: object) -> "Evaluator":
        _ensure_registry_initialized()
        evaluator = _REGISTRY.create(name, **kwargs)
        return cls(evaluator=evaluator, name=name)

    @classmethod
    def from_function(cls, func: MetricFunction, *, name: str | None = None) -> "Evaluator":
        class FunctionAdapter(IEvaluator):
            def __init__(self, func: MetricFunction) -> None:
                self.func = func

            def evaluate(
                self, system_output: object, correct_answer: object, **kwargs: object
            ) -> EvaluationResult:
                metrics = self.func(system_output, correct_answer, **kwargs)

                if "is_correct" not in metrics:
                    raise KeyError("Metric functions must return an 'is_correct' key")
                raw_is_correct = metrics["is_correct"]
                if isinstance(raw_is_correct, bool):
                    is_correct = raw_is_correct
                elif isinstance(raw_is_correct, (int, float)) and raw_is_correct in (0, 1):
                    is_correct = bool(raw_is_correct)
                else:
                    raise TypeError(
                        "Metric functions must return 'is_correct' as bool "
                        f"(or 0/1), got {raw_is_correct!r}"
                    )

                raw_score = metrics.get("score")
                if raw_score is None:
                    score = 1.0 if is_correct else 0.0
                elif isinstance(raw_score, bool):
                    score = 1.0 if raw_score else 0.0
                elif isinstance(raw_score, (int, float)):
                    score = float(raw_score)
                else:
                    raise TypeError(
                        "Metric functions must return 'score' as a number, "
                        f"got {raw_score!r}"
                    )

                return EvaluationResult(is_correct=is_correct, score=score, metadata=metrics)

        resolved_name = name or getattr(func, "__name__", "metric")
        return cls(evaluator=FunctionAdapter(func), name=resolved_name)

    def __init__(self, evaluator: IEvaluator, *, name: str = "evaluator") -> None:
        self.evaluator = evaluator
        self.name = name

    def evaluate(
        self, prediction: object, reference: object, **kwargs: object
    ) -> dict[str, object]:
        result = self.evaluator.evaluate(prediction, reference, **kwargs)

        metrics: dict[str, object] = {}
        metrics["is_correct"] = result.is_correct
        metrics["score"] = result.score
        if result.metadata:
            metrics.update(
                {
                    key: value
                    for key, value in result.metadata.items()
                    if key not in {"is_correct", "score"}
                }
            )

        return metrics


class EvaluationPipeline:
    """Utility that applies multiple evaluators across an iterable dataset."""

    def __init__(
        self,
        evaluators: list[Evaluator],
        *,
        input_key: str = "question",
        reference_key: str = "answer",
    ) -> None:
        self.evaluators = evaluators
        self.input_key = input_key
        self.reference_key = reference_key

    def _unique_evaluator_names(self) -> list[str]:
        counts: dict[str, int] = {}
        resolved: list[str] = []
        for evaluator in self.evaluators:
            base = evaluator.name or evaluator.evaluator.__class__.__name__
            seen = counts.get(base, 0) + 1
            counts[base] = seen
            resolved.append(base if seen == 1 else f"{base}_{seen}")
        return resolved

    def evaluate(
        self,
        model: Callable[[str], object],
        dataset: Iterable[Mapping[str, object]],
    ) -> dict[str, NumericMetric]:
        all_metrics: dict[str, list[float]] = {}
        processed_count = 0
        evaluator_names = self._unique_evaluator_names()
        namespace_metrics = len(self.evaluators) > 1

        for idx, row in enumerate(dataset):
            processed_count += 1
            try:
                input_value = row[self.input_key]
            except KeyError as exc:
                available = ", ".join(sorted(row.keys()))
                raise KeyError(
                    f"Dataset row {idx} missing input_key={self.input_key!r}. "
                    f"Available keys: {available}"
                ) from exc

            if not isinstance(input_value, str):
                raise TypeError(
                    f"Dataset row {idx} input_key={self.input_key!r} must be str; "
                    f"got {type(input_value).__name__}"
                )

            try:
                reference = row[self.reference_key]
            except KeyError as exc:
                available = ", ".join(sorted(row.keys()))
                raise KeyError(
                    f"Dataset row {idx} missing reference_key={self.reference_key!r}. "
                    f"Available keys: {available}"
                ) from exc

            try:
                prediction = model(input_value)
            except Exception as exc:
                raise RuntimeError(f"Model failed for dataset row {idx}") from exc

            for evaluator_name, evaluator in zip(
                evaluator_names,
                self.evaluators,
                strict=True,
            ):
                try:
                    metrics = evaluator.evaluate(prediction, reference)
                except Exception as exc:
                    raise RuntimeError(
                        f"Evaluator {evaluator_name!r} failed for dataset row {idx}"
                    ) from exc

                for key, value in metrics.items():
                    if isinstance(value, bool):
                        metric_value = 1.0 if value else 0.0
                    elif isinstance(value, (int, float)):
                        metric_value = float(value)
                    else:
                        continue

                    metric_key = f"{evaluator_name}.{key}" if namespace_metrics else key
                    all_metrics.setdefault(metric_key, []).append(metric_value)

        results: dict[str, NumericMetric] = {}
        for k, v in all_metrics.items():
            if v:  # Non-empty list
                results[k] = sum(v) / len(v)
            else:
                results[k] = 0.0

        results["processed_count"] = processed_count

        return results


__all__ = [
    "Evaluator",
    "EvaluationPipeline",
    "list_available_evaluators",
    "register_evaluator",
    "EvaluationResult",
    "IEvaluator",
]
