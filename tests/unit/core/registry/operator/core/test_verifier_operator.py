from ember.core.registry.operator.core.verifier import (
    VerifierOperator,
    VerifierOperatorInputs,
    VerifierOperatorOutputs,
)


class CustomVerifierLM:
    """Mimics a verifier that outputs verdict, explanation, optional revised answer lines."""

    def __call__(self, *, prompt: str) -> str:
        """Dummy LM __call__ that returns a verification output."""
        del prompt
        return (
            "Verdict: 1\n"
            "Explanation: The answer is correct because...\n"
            "Revised Answer: \n"
        )


def test_verifier_operator_forward() -> None:
    custom_lm = CustomVerifierLM()
    op = VerifierOperator(lm_module=custom_lm)

    inputs = VerifierOperatorInputs(query="Verify this", candidate_answer="Answer")
    result: VerifierOperatorOutputs = op(inputs=inputs)

    # Verdict is numeric: 1 means correct.
    assert (
        result.verdict == 1
    ), "VerifierOperator did not return the expected verdict (1 for correct)."
    assert (
        result.explanation == "The answer is correct because..."
    ), "VerifierOperator did not return the expected explanation."
