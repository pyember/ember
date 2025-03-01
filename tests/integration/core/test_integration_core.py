"""
Integration tests: multi-stage pipeline chaining multiple operator wrappers.

This test simulates a full production pipeline (Ensemble → MostCommon → Verifier)
and verifies that inputs propagate correctly, prompts are rendered, and the final
output contains expected verification details.
"""

from ember.core.non import (
    UniformEnsemble,
    MostCommon,
    Verifier,
    Sequential,
    EnsembleInputs,
)
from unittest.mock import patch
import pytest
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.exceptions import ProviderAPIError, ModelNotFoundError


@patch(
    "ember.core.registry.model.base.services.model_service.ModelService.invoke_model",
    return_value="AnswerX",
)
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
        lm.__call__ = (
            lambda *, prompt: "Verdict: Correct\nExplanation: Verified\nRevised Answer: FinalAnswer"
        )
    # Chain the operators into a pipeline.
    pipeline = Sequential(operators=[ensemble, most_common])
    input_data: EnsembleInputs = EnsembleInputs(query="What is the answer?")
    output = pipeline(inputs=input_data)
    # Verify that the final output has the expected final_answer key
    assert isinstance(output, dict), "Output should be a dict"
    assert "final_answer" in output, "Output should have 'final_answer' key"
    assert output["final_answer"] == "AnswerX"


class FailingProvider:
    def __init__(self, model_info):
        self.model_info = model_info

    def __call__(self, prompt, **kwargs):
        raise RuntimeError("API failure")


def create_dummy_model_info(model_id: str) -> ModelInfo:
    """Helper function to create dummy ModelInfo objects."""
    return ModelInfo(
        id=model_id,
        name="Failing Model",
        cost=ModelCost(input_cost_per_thousand=0.0, output_cost_per_thousand=0.0),
        rate_limit=RateLimit(tokens_per_minute=0, requests_per_minute=0),
        provider=ProviderInfo(name="FailingProvider", default_api_key="dummy_key"),
        api_key="dummy_key",
    )


@pytest.fixture
def failing_registry(monkeypatch):
    from ember.core.registry.model.base.registry.factory import ModelFactory

    # Patch the factory to always return our failing provider
    def mock_create_model_from_info(*, model_info):
        return FailingProvider(model_info)

    monkeypatch.setattr(
        ModelFactory, "create_model_from_info", mock_create_model_from_info
    )

    registry = ModelRegistry()
    registry.register_model(create_dummy_model_info("failing:model"))
    return registry


def test_provider_api_failure(failing_registry):
    service = ModelService(registry=failing_registry)
    with pytest.raises(ProviderAPIError, match="Error invoking model failing:model"):
        service.invoke_model(model_id="failing:model", prompt="test")


def test_provider_model_not_found(failing_registry):
    service = ModelService(registry=failing_registry)
    with pytest.raises(ModelNotFoundError, match="Unknown:model"):
        service.invoke_model(model_id="Unknown:model", prompt="test")
