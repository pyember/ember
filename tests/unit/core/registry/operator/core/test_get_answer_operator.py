import pytest
from typing import Dict, Any, List

from src.ember.core.registry.operator.core.get_answer import (
    GetAnswerOperator,
    GetAnswerOperatorInputs,
)
from src.ember.core.registry.prompt_signature.signatures import Signature


class DummyLMModule:
    """Stand-in LM module that echoes the prompt."""

    def __call__(self, *, prompt: str) -> str:
        return f"Echo: {prompt}"


def test_get_answer_operator_forward() -> None:
    dummy_lm = DummyLMModule()
    op = GetAnswerOperator(lm_module=dummy_lm)

    inputs = GetAnswerOperatorInputs(query="What is 2+2?", response="The answer is 4.")
    result: Dict[str, Any] = op(inputs=inputs)

    # Validate the final answer matches what the LM returns
    prompt_text = op.signature.render_prompt(inputs=inputs.model_dump())
    expected_answer = f"Echo: {prompt_text}"
    assert (
        result.get("final_answer") == expected_answer
    ), "GetAnswerOperator did not use the LM module output correctly."
