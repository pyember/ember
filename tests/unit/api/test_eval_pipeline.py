from __future__ import annotations

import pytest

from ember.api.eval import EvaluationPipeline, Evaluator
from ember.utils.eval.base_evaluator import EvaluationResult, IEvaluator


class _StaticEvaluator(IEvaluator[str, str]):
    def __init__(self, result: EvaluationResult) -> None:
        self._result = result

    def evaluate(
        self, system_output: str, correct_answer: str, **kwargs: object
    ) -> EvaluationResult:
        return self._result


def test_pipeline_single_evaluator_keeps_flat_metrics() -> None:
    pipeline = EvaluationPipeline([Evaluator.from_registry("exact_match")])
    dataset = [{"question": "Capital?", "answer": "Paris"}]

    result = pipeline.evaluate(lambda _: "Paris", dataset)

    assert result["is_correct"] == 1.0
    assert result["score"] == 1.0
    assert result["processed_count"] == 1


def test_pipeline_multiple_evaluators_namespaces_metrics() -> None:
    pipeline = EvaluationPipeline(
        [
            Evaluator.from_registry("exact_match"),
            Evaluator.from_registry("accuracy"),
        ]
    )
    dataset = [{"question": "Capital?", "answer": "Paris"}]

    result = pipeline.evaluate(lambda _: "Paris", dataset)

    assert "is_correct" not in result
    assert "score" not in result
    assert result["exact_match.is_correct"] == 1.0
    assert result["accuracy.is_correct"] == 1.0


def test_pipeline_raises_with_context_on_model_error() -> None:
    pipeline = EvaluationPipeline([Evaluator.from_registry("exact_match")])
    dataset = [
        {"question": "ok", "answer": "ok"},
        {"question": "boom", "answer": "boom"},
    ]

    def model(query: str) -> str:
        if query == "boom":
            raise RuntimeError("boom")
        return query

    with pytest.raises(RuntimeError, match="Model failed for dataset row 1"):
        pipeline.evaluate(model, dataset)


def test_evaluator_does_not_allow_metadata_overrides() -> None:
    evaluator = Evaluator(
        evaluator=_StaticEvaluator(
            EvaluationResult(
                is_correct=True,
                score=1.0,
                metadata={"is_correct": False, "score": 0.0, "extra": 1},
            )
        ),
        name="static",
    )

    assert evaluator.evaluate("x", "y") == {"is_correct": True, "score": 1.0, "extra": 1}


def test_metric_function_rejects_invalid_is_correct_type() -> None:
    def bad_metric(_prediction: str, _reference: str, **_kwargs: object) -> dict[str, object]:
        return {"is_correct": "yes"}  # type: ignore[return-value]

    evaluator = Evaluator.from_function(bad_metric)
    with pytest.raises(TypeError, match="must return 'is_correct' as bool"):
        evaluator.evaluate("x", "y")
