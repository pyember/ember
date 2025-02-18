from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List
from pydantic import BaseModel

from src.ember.core.registry.operator.base.operator_base import Operator

from src.ember.core.registry.prompt_signature.signatures import Signature


class MostCommonOperatorInputs(BaseModel):
    """Input model for MostCommonOperator.

    Attributes:
        responses (List[str]): A list of response strings.
    """

    responses: List[str]


class MostCommonOperator(Operator[MostCommonOperatorInputs, Dict[str, Any]]):
    """Operator that selects the most common answer from provided responses."""

    signature: Signature = Signature(input_model=MostCommonOperatorInputs)

    def forward(self, *, inputs: MostCommonOperatorInputs) -> Dict[str, Any]:
        if not inputs.responses:
            return {"final_answer": None}
        counts: Counter = Counter(inputs.responses)
        most_common_answer: str = counts.most_common(1)[0][0]
        return {"final_answer": most_common_answer}
