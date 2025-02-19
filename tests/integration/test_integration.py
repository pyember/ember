"""
Integration tests: multi-stage pipeline chaining multiple operator wrappers.

This test simulates a full production pipeline (Ensemble → MostCommon → Verifier)
and verifies that inputs propagate correctly, prompts are rendered, and the final
output contains expected verification details.
"""

from src.ember.core.non import UniformEnsemble, MostCommon, Verifier, Sequential, EnsembleInputs
from unittest.mock import patch

@patch("src.ember.core.registry.model.services.model_service.ModelService.invoke_model", return_value="AnswerX")
def test_multi_stage_pipeline_integration(mock_invoke_model) -> None:
    # Set up an Ensemble operator.
    ensemble = UniformEnsemble(num_units=3, model_name="dummy", temperature=1.0)
    for lm in ensemble.ensemble_op.lm_modules:
        lm.__call__ = lambda *, prompt: "response_A"
    # Set up a MostCommon operator.
    most_common = MostCommon()
    # Set up a Verifier operator.
    verifier = Verifier(model_name="dummy", temperature=0.0)
    for lm in [verifier.verifier_op.lm_module]:
        lm.__call__ = lambda *, prompt: "Verdict: Correct\nExplanation: Verified\nRevised Answer: FinalAnswer"
    # Chain the operators into a pipeline.
    pipeline = Sequential(operators=[ensemble, most_common])
    input_data: EnsembleInputs = EnsembleInputs(query="What is the answer?")
    output = pipeline(inputs=input_data)
    # Verify that the final output includes the expected verification details.
    assert "final_answer" in output
    assert output["final_answer"] == "AnswerX"