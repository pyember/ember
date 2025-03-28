from __future__ import annotations

from collections import Counter
from typing import List, Optional

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types import EmberModel


class MostCommonAnswerSelectorOperatorInputs(EmberModel):
    """Input model for MostCommonAnswerSelectorOperator.

    Attributes:
        responses (List[str]): A list of response strings.
    """

    responses: List[str]


class MostCommonAnswerSelectorOutputs(EmberModel):
    """Output model for MostCommonAnswerSelectorOperator.

    Attributes:
        final_answer (Optional[str]): The most common answer from responses.
    """

    final_answer: Optional[str]


class MostCommonAnswerSelectorOperator(
    Operator[MostCommonAnswerSelectorOperatorInputs, MostCommonAnswerSelectorOutputs]
):
    """Operator that selects the most common answer from provided responses."""

    specification: Specification = Specification(
        input_model=MostCommonAnswerSelectorOperatorInputs,
        structured_output=MostCommonAnswerSelectorOutputs,
    )

    def forward(
        self, *, inputs: MostCommonAnswerSelectorOperatorInputs
    ) -> MostCommonAnswerSelectorOutputs:
        if not inputs.responses:
            return {"final_answer": None}

        counts: Counter = Counter(inputs.responses)

        # Handle potential empty responses
        if not counts:
            return {"final_answer": None}

        most_common_answer: str = counts.most_common(1)[0][0]
        return {"final_answer": most_common_answer}
