import pytest

from ember.core.registry.operator.core.ensemble import (
    EnsembleOperator,
    EnsembleOperatorInputs,
    EnsembleOperatorOutputs,
)
from ember.core.registry.prompt_signature.signatures import Signature


class DummyLMModule:
    """Simple mock LM module that returns a standardized response."""

    def __call__(self, *, prompt: str) -> str:
        return f"LM response to: {prompt}"


def test_ensemble_operator_forward() -> None:
    dummy_lm1 = DummyLMModule()
    dummy_lm2 = DummyLMModule()

    # Optionally customize the operator's signature:
    custom_signature = Signature(
        input_model=EnsembleOperatorInputs, prompt_template="Ensemble Prompt: {query}"
    )

    op = EnsembleOperator(lm_modules=[dummy_lm1, dummy_lm2])
    # Override the default signature:
    op.signature = custom_signature

    inputs = EnsembleOperatorInputs(query="test query")
    result: EnsembleOperatorOutputs = op(inputs=inputs)

    # Verify the aggregated responses:
    rendered_prompt = custom_signature.render_prompt(inputs=inputs)
    expected_responses = [
        dummy_lm1(prompt=rendered_prompt),
        dummy_lm2(prompt=rendered_prompt),
    ]

    assert isinstance(result, dict), "Result should be a dict (which will be converted to EnsembleOperatorOutputs by the framework)"
    assert "responses" in result, "Result should contain 'responses' key"
    assert result["responses"] == expected_responses, "Responses should match expected responses"
