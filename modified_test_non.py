"""
Modified tests for NON wrappers with better mocking.

This file tests the wrappers with more focused and isolated mocking, to ensure
that we can test functionality without complex dependencies like the model registry.
"""

import pytest
import sys
from typing import Any, Dict, List, Optional, Type, Union
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

# First ensure the path is correct
PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# Now we can import from ember
from ember.core.non import (
    UniformEnsemble,
    MostCommon,
    JudgeSynthesis,
    Verifier,
    VariedEnsemble,
    Sequential,
    EnsembleInputs,
    MostCommonInputs,
    JudgeSynthesisInputs,
    VerifierInputs,
    VerifierOutputs,
    VariedEnsembleInputs,
    VariedEnsembleOutputs,
)

# Mock model and registry components
class MockModel:
    """Mock model for tests."""
    
    def __init__(self, model_id: str = "dummy"):
        self.model_id = model_id
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Mock generate method."""
        return "Default mock response"


class MockRegistry:
    """Mock model registry."""
    
    def __init__(self):
        self.models = {"dummy": MockModel()}
        
    def get_model(self, model_id: str) -> MockModel:
        """Mock get_model method."""
        return self.models.get(model_id, MockModel())


# Helper functions for testing
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
    return "Verdict: 1\nExplanation: All good\nRevised Answer: AnswerY"


def failing_lm(*, prompt: str) -> str:
    """Dummy LM __call__ that always raises an exception."""
    raise Exception("Simulated LM failure")


# ------------------------------------------------------------------------------
# 1) MostCommon Operator Tests
# ------------------------------------------------------------------------------
def test_most_common_operator_normal() -> None:
    """Test that MostCommon returns the most common answer from candidate responses."""
    aggregator = MostCommon()
    inputs: MostCommonInputs = MostCommonInputs(
        query="Test", responses=["A", "B", "A", "C", "A"]
    )
    output: Dict[str, Any] = aggregator.forward(inputs=inputs)
    assert "final_answer" in output, "Output should include 'final_answer'."
    assert output["final_answer"] == "A", "The most common answer should be 'A'."


# ------------------------------------------------------------------------------
# 2) Sequential Pipeline Tests
# ------------------------------------------------------------------------------
def test_sequential_pipeline_operator_normal() -> None:
    """Test that Sequential chains operators in order."""
    from ember.core.registry.operator.base.operator_base import Operator
    from ember.core.registry.prompt_signature.signatures import Signature

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
# 3) VariedEnsemble Operator Tests
# ------------------------------------------------------------------------------

@patch(
    "ember.core.registry.model.model_module.lm.LMModule.__call__",
    return_value="varied response",
)
def test_varied_ensemble_operator_normal(mock_call) -> None:
    """Test that VariedEnsemble aggregates responses from multiple LM configurations."""
    from ember.core.registry.model.model_module.lm import LMModuleConfig

    dummy_config: LMModuleConfig = LMModuleConfig(model_id="dummy", temperature=1.0)
    varied_ensemble = VariedEnsemble(model_configs=[dummy_config, dummy_config])
    
    # Test with mocked method
    inputs: VariedEnsembleInputs = VariedEnsembleInputs(query="Test")
    outputs: VariedEnsembleOutputs = varied_ensemble(inputs=inputs)
    
    assert isinstance(outputs, VariedEnsembleOutputs), "Output should be a VariedEnsembleOutputs."
    assert isinstance(outputs.responses, list), "The responses should be a list."
    assert len(outputs.responses) == 2, "There should be 2 responses."
    assert all(r == "varied response" for r in outputs.responses), "Each response should be correct."


if __name__ == "__main__":
    pytest.main(["modified_test_non.py", "-v"])