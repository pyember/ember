from collections import Counter

from tests.helpers.simplified_imports import EmberModel


# Create test stubs
class MostCommonAnswerSelectorOperatorInputs(EmberModel):
    """Test input model."""

    responses: list[str]


class MostCommonAnswerSelectorOutputs(EmberModel):
    """Test output model."""

    final_answer: str = None


class MostCommonAnswerSelectorOperator:
    """Test operator implementation."""

    def __init__(self):
        pass

    def __call__(self, *, inputs):
        return self.forward(inputs=inputs)

    def forward(self, *, inputs):
        """Select the most common answer."""
        if not inputs.responses:
            return {"final_answer": None}

        counts = Counter(inputs.responses)
        if not counts:
            return {"final_answer": None}

        most_common_answer = counts.most_common(1)[0][0]
        return {"final_answer": most_common_answer}


def test_most_common_operator_forward() -> None:
    inputs = MostCommonAnswerSelectorOperatorInputs(responses=["A", "B", "A", "C"])
    op = MostCommonAnswerSelectorOperator()
    result: MostCommonAnswerSelectorOutputs = op(inputs=inputs)
    assert (
        result["final_answer"] == "A"
    ), "MostCommonOperator did not return the most common answer."
