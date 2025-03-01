import pytest
from typing import Dict

from ember.core.registry.operator.core.most_common import (
    MostCommonAnswerSelectorOperator,
    MostCommonAnswerSelectorOperatorInputs,
    MostCommonAnswerSelectorOutputs,
)


def test_most_common_operator_forward() -> None:
    inputs = MostCommonAnswerSelectorOperatorInputs(
        query="dummy query", responses=["A", "B", "A", "C"]
    )
    op = MostCommonAnswerSelectorOperator()
    result: MostCommonAnswerSelectorOutputs = op(inputs=inputs)
    assert (
        result["final_answer"] == "A"
    ), "MostCommonOperator did not return the most common answer."
