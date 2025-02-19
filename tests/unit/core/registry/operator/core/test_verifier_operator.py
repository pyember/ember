import pytest
from typing import Dict, Any

from src.ember.core.registry.operator.core.verifier import (
    VerifierOperator,
    VerifierOperatorInputs
)


class CustomVerifierLM:
    """Mimics a verifier that outputs verdict, explanation, optional revised answer lines."""
    def __call__(self, *, prompt: str) -> str:
        return (
            "Verdict: Correct\n"
            "Explanation: The answer is correct because...\n"
            "Revised Answer: \n"
        )


def test_verifier_operator_forward() -> None:
    custom_lm = CustomVerifierLM()
    op = VerifierOperator(lm_module=custom_lm)

    inputs = VerifierOperatorInputs(query="Verify this", candidate_answer="Answer")
    result: Dict[str, Any] = op(inputs=inputs)

    assert result.get("verdict") == "Correct", "VerifierOperator did not return the expected verdict."
    assert result.get("explanation") == "The answer is correct because...", (
        "VerifierOperator did not return the expected explanation."
    ) 