from __future__ import annotations

import pytest

from ember.utils.eval.numeric_answer import AIMEAnswerEvaluator


@pytest.fixture()
def evaluator() -> AIMEAnswerEvaluator:
    return AIMEAnswerEvaluator()


def test_hash_format_answer(evaluator: AIMEAnswerEvaluator) -> None:
    response = "Work shown above\n\n### 150"
    result = evaluator.evaluate(response, "150")

    assert result.is_correct
    assert result.score == 1.0
    assert result.metadata["extracted_method"] == "final_pattern"
    assert result.metadata["extractor"] == "HashAnswerExtractor"


def test_hash_format_with_quotes(evaluator: AIMEAnswerEvaluator) -> None:
    response = '"After the derivation we reach the final value."\n### 033'
    result = evaluator.evaluate(response, "033")

    assert result.is_correct
    assert result.metadata["extractor"] == "HashAnswerExtractor"
