from typing import Any, Dict

from tests.helpers.simplified_imports import EmberModel


# Create test stubs
class JudgeSynthesisInputs(EmberModel):
    """Test input model."""

    query: str
    responses: list[str]


class JudgeSynthesisOutputs(EmberModel):
    """Test output model."""

    final_answer: str
    reasoning: str


class JudgeSynthesisOperator:
    """Test operator implementation."""

    def __init__(self, *, lm_module):
        self.lm_module = lm_module

    def __call__(self, *, inputs):
        return self.forward(inputs=inputs)

    def forward(self, *, inputs):
        """Process the synthesis."""
        raw_output = self.lm_module(
            prompt=f"Query: {inputs.query}\nResponses: {inputs.responses}"
        )

        # Parse the response
        final_answer = "Unknown"
        reasoning = ""

        for line in raw_output.splitlines():
            if line.startswith("Final Answer:"):
                final_answer = line.replace("Final Answer:", "").strip()
            elif line.startswith("Reasoning:"):
                reasoning = line.replace("Reasoning:", "").strip()

        return JudgeSynthesisOutputs(final_answer=final_answer, reasoning=reasoning)


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
