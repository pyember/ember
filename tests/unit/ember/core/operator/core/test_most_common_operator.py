import pytest
from typing import Dict

from src.ember.core.registry.operator.core.most_common import (
    MostCommonOperator,
    MostCommonOperatorInputs
)


def test_most_common_operator_forward() -> None:
    inputs = MostCommonOperatorInputs(
        query="dummy query",
        responses=["A", "B", "A", "C"]
    )
    op = MostCommonOperator()
    result: Dict[str, str] = op(inputs=inputs)
    assert result.get("final_answer") == "A", "MostCommonOperator did not return the most common answer." 