#!/usr/bin/env python3
"""
Tests for NON wrappers.

These tests verify that each wrapper:
  - Returns the expected output when given valid inputs.
  - Propagates errors when a dummy LM module fails.
  - Supports sequential composition via a pipeline.

This file is production grade, strongly typed, and follows the Google Python Style Guide.
"""

from typing import Any, Dict, List
import pytest
from pydantic import BaseModel, Field
from unittest.mock import patch
from src.ember.core.registry.model.services.model_service import ModelService

# Import the wrappers and their input/output types from non.py.
from src.ember.core.non import (
    UniformEnsemble,
    MostCommon,
    GetAnswer,
    JudgeSynthesis,
    Verifier,
    VariedEnsemble,
    Sequential,
    EnsembleInputs,
    MostCommonInputs,
    GetAnswerInputs,
    JudgeSynthesisInputs,
    VerifierInputs,
    VariedEnsembleInputs,
    VariedEnsembleOutputs,
)
# For dummy operator in sequential test.
from src.ember.core.registry.operator.base.operator_base import Operator
from src.ember.core.registry.prompt_signature.signatures import Signature


# ------------------------------------------------------------------------------
# Helper: Dummy LM behavior
# ------------------------------------------------------------------------------

def dummy_response_lm(*, prompt: str) -> str:
    """Dummy LM __call__ that always returns a fixed response."""
    return "response"

def dummy_final_answer_lm(*, prompt: str) -> str:
    """Dummy LM __call__ that returns a final answer string."""
    return "Final Answer: AnswerX"

def dummy_reasoning_lm(*, prompt: str) -> str:
    """Dummy LM __call__ that returns reasoning and a final answer."""
    return "Reasoning: Some reasoning\nFinal Answer: Synthesized"

def dummy_verifier_lm(*, prompt: str) -> str:
    """Dummy LM __call__ that returns a verification output."""
    return (
        "Verdict: Correct\n"
        "Explanation: All good\n"
        "Revised Answer: AnswerY"
    )

def failing_lm(*, prompt: str) -> str:
    """Dummy LM __call__ that always raises an exception."""
    raise Exception("Simulated LM failure")


# ------------------------------------------------------------------------------
# 1) UniformEnsemble Operator Tests
# ------------------------------------------------------------------------------

@patch("src.ember.core.registry.model.services.model_service.ModelService.invoke_model", return_value="response")
def test_uniform_ensemble_operator_normal(mock_invoke):
    """Test that UniformEnsemble returns a dictionary with responses from multiple LM modules."""
    uniform_ensemble = UniformEnsemble(num_units=3, model_name="dummy", temperature=1.0)
    # Override __call__ on each LM module to return a fixed response.
    for lm in uniform_ensemble.ensemble_op.lm_modules:
        lm.__call__ = dummy_response_lm  # type: ignore
    inputs: EnsembleInputs = EnsembleInputs(query="Test query")
    output: Dict[str, Any] = uniform_ensemble.forward(inputs=inputs)
    assert isinstance(output, dict), "Output should be a dict."
    assert "responses" in output, "Output dict should contain 'responses'."
    responses = output["responses"]
    assert isinstance(responses, list), "'responses' should be a list."
    assert len(responses) == 3, "There should be 3 responses."
    for resp in responses:
        print('resp: ', resp)
        assert resp == "response", "Each response should be 'response'."


# ------------------------------------------------------------------------------
# 2) MostCommon Operator Tests
# ------------------------------------------------------------------------------

def test_most_common_operator_normal() -> None:
    """Test that MostCommon returns the most common answer from candidate responses."""
    aggregator = MostCommon()
    inputs: MostCommonInputs = MostCommonInputs(query="Test", responses=["A", "B", "A", "C", "A"])
    output: Dict[str, Any] = aggregator.forward(inputs=inputs)
    assert "final_answer" in output, "Output should include 'final_answer'."
    assert output["final_answer"] == "A", "The most common answer should be 'A'."


# ------------------------------------------------------------------------------
# 3) GetAnswer Operator Tests
# ------------------------------------------------------------------------------

@patch("src.ember.core.registry.model.services.model_service.ModelService.invoke_model", return_value="AnswerX")
def test_get_answer_operator_normal(mock_invoke) -> None:
    """Test that GetAnswer extracts the final answer from LM module output."""
    getter = GetAnswer(model_name="dummy", temperature=0.0)
    inputs: GetAnswerInputs = GetAnswerInputs(query="Test", responses=["ignored"])
    output: Dict[str, Any] = getter(inputs=inputs)
    assert "final_answer" in output, "Output should contain 'final_answer'."
    assert output["final_answer"] == "AnswerX", "The final answer should be 'AnswerX'."


# ------------------------------------------------------------------------------
# 4) JudgeSynthesis Operator Tests
# ------------------------------------------------------------------------------

@patch("src.ember.core.registry.model.services.model_service.ModelService.invoke_model", return_value="Reasoning: Some reasoning\nFinal Answer: Synthesized")
def test_judge_synthesis_operator_normal(mock_invoke) -> None:
    """Test that JudgeSynthesis synthesizes a final answer and reasoning."""
    judge = JudgeSynthesis(model_name="dummy", temperature=1.0)
    # Override __call__ on LM module(s) to simulate LM output with reasoning.
    for lm in [judge.judge_synthesis_op.lm_module]:
        lm.__call__ = dummy_reasoning_lm  # type: ignore
    inputs: JudgeSynthesisInputs = JudgeSynthesisInputs(query="Test", responses=["Resp1", "Resp2"])
    output: Dict[str, Any] = judge(inputs=inputs)
    assert "final_answer" in output, "Output must include 'final_answer'."
    assert "reasoning" in output, "Output must include 'reasoning'."
    assert output["final_answer"] == "Synthesized", "Final answer should be 'Synthesized'."
    assert "Some reasoning" in output["reasoning"], "Reasoning should contain 'Some reasoning'."


# ------------------------------------------------------------------------------
# 5) Verifier Operator Tests
# ------------------------------------------------------------------------------

@patch("src.ember.core.registry.model.services.model_service.ModelService.invoke_model", return_value="Verdict: Correct\nExplanation: All good\nRevised Answer: AnswerY")
def test_verifier_operator_normal(mock_invoke) -> None:
    """Test that Verifier returns a verdict, explanation, and revised answer."""
    verifier = Verifier(model_name="dummy", temperature=1.0)
    # Override __call__ on LM module(s) to simulate a fixed verification response.
    for lm in [verifier.verifier_op.lm_module]:
        lm.__call__ = dummy_verifier_lm  # type: ignore
    inputs: VerifierInputs = VerifierInputs(query="Test", candidate_answer="AnswerX")
    output: Dict[str, Any] = verifier(inputs=inputs)
    assert "verdict" in output, "Output must include 'verdict'."
    assert "explanation" in output, "Output must include 'explanation'."
    assert "revised_answer" in output, "Output must include 'revised_answer'."
    assert output["verdict"] == "Correct", "Verdict should be 'Correct'."
    assert output["revised_answer"] == "AnswerY", "Revised answer should be 'AnswerY'."


# ------------------------------------------------------------------------------
# 6) VariedEnsemble Operator Tests
# ------------------------------------------------------------------------------

@patch("src.ember.core.registry.model.services.model_service.ModelService.invoke_model", return_value="varied response")
def test_varied_ensemble_operator_normal(mock_invoke) -> None:
    """Test that VariedEnsemble aggregates responses from multiple LM configurations."""
    from src.ember.core.registry.model.modules.lm import LMModuleConfig
    dummy_config: LMModuleConfig = LMModuleConfig(model_id="dummy", temperature=1.0)
    varied_ensemble = VariedEnsemble(model_configs=[dummy_config, dummy_config])
    # Override call_lm to return a fixed response.
    varied_ensemble.call_lm = lambda *, prompt, lm: "varied response"  # type: ignore
    inputs: VariedEnsembleInputs = VariedEnsembleInputs(query="Test")
    outputs: VariedEnsembleOutputs = varied_ensemble(inputs=inputs)
    assert isinstance(outputs, VariedEnsembleOutputs), "Output should be an instance of VariedEnsembleOutputs."
    assert isinstance(outputs.responses, list), "The responses should be a list."
    assert len(outputs.responses) == 2, "There should be 2 responses."
    for resp in outputs.responses:
        assert resp == "varied response", "Each response should be 'varied response'."


# ------------------------------------------------------------------------------
# 7) Sequential Pipeline Tests
# ------------------------------------------------------------------------------

def test_sequential_pipeline_operator_normal() -> None:
    """Test that Sequential chains operators in order."""
    
    # Define a dummy operator that increments a value.
    class DummyOp(Operator[Dict[str, Any], Dict[str, Any]]):
        # Provide a dummy signature so that __call__ does not fail.
        signature: Signature = Signature(input_model=None)

        def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
            # Increment the value by 1.
            return {"value": inputs.get("value", 0) + 1}
    
    # Instantiate two dummy operators.
    op1 = DummyOp()
    op2 = DummyOp()
    pipeline = Sequential(operators=[op1, op2])
    output: Dict[str, Any] = pipeline(inputs={"value": 0})
    assert isinstance(output, dict), "Output must be a dict."
    assert output.get("value") == 2, "Value should be incremented twice to 2."


# ------------------------------------------------------------------------------
# 8) Error Propagation Tests
# ------------------------------------------------------------------------------

@patch("src.ember.core.registry.model.services.model_service.ModelService.invoke_model", side_effect=Exception("Simulated LM failure"))
def test_uniform_ensemble_operator_failure_propagation(mock_invoke) -> None:
    """Test that UniformEnsemble propagates errors when an LM module fails."""
    uniform_ensemble = UniformEnsemble(num_units=1, model_name="dummy", temperature=1.0)
    # Override __call__ on each LM module to simulate a failure.
    for lm in uniform_ensemble.ensemble_op.lm_modules:
        lm.__call__ = failing_lm  # type: ignore
    inputs: EnsembleInputs = EnsembleInputs(query="Test")
    with pytest.raises(Exception, match="Simulated LM failure"):
        uniform_ensemble(inputs=inputs)


if __name__ == "__main__":
    pytest.main()