import pytest
from typing import Dict, Any

from ember.core.registry.operator.core.synthesis_judge import (
    JudgeSynthesisOperator,
    JudgeSynthesisInputs,
)


class CustomLMModule:
    """Returns reasoning plus a final answer line."""

    def __call__(self, *, prompt: str) -> str:
        return "Reasoning: Some reasoning here.\nFinal Answer: Synthesized Answer"


def test_judge_synthesis_operator_forward() -> None:
    custom_lm = CustomLMModule()
    op = JudgeSynthesisOperator(lm_module=custom_lm)

    inputs = JudgeSynthesisInputs(query="synthesize?", responses=["Ans1", "Ans2"])
    result: Dict[str, Any] = op(inputs=inputs)

    assert (
        result.final_answer == "Synthesized Answer"
    ), "JudgeSynthesisOperator did not synthesize the expected final answer."
    assert (
        "Some reasoning here." in result.reasoning
    ), "JudgeSynthesisOperator did not capture reasoning correctly."
