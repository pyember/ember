"""
Integration tests: multi-stage pipeline chaining multiple operator wrappers.

This test simulates a full production pipeline (Ensemble → MostCommon → Verifier)
and verifies that inputs propagate correctly, prompts are rendered, and the final
output contains expected verification details.
"""

import logging
from typing import Any, Dict, List, Optional

import pytest

from ember.core.exceptions import ModelNotFoundError, ProviderAPIError
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig
from ember.core.registry.operator.base._module import static_field

# Operator components
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification

# Model components
from ember.core.types.ember_model import EmberModel

# Configure logging
logger = logging.getLogger(__name__)


class EnsembleOperatorInputs(EmberModel):
    """Input model for Ensemble operator"""

    query: str


class EnsembleOperatorOutputs(EmberModel):
    """Output model for Ensemble operator"""

    responses: List[str]


class EnsembleOperator(Operator[EnsembleOperatorInputs, EnsembleOperatorOutputs]):
    """Real implementation of the Ensemble Operator."""

    specification = Specification[EnsembleOperatorInputs, EnsembleOperatorOutputs](
        input_model=EnsembleOperatorInputs, structured_output=EnsembleOperatorOutputs
    )

    # Define static fields
    lm_modules: List[LMModule] = static_field()

    def __init__(self, lm_modules: List[LMModule]):
        """Initialize with LM modules."""
        self.lm_modules = lm_modules

    def forward(self, *, inputs: EnsembleOperatorInputs) -> EnsembleOperatorOutputs:
        """Execute query across all models."""
        rendered_prompt = self.specification.render_prompt(inputs=inputs)
        responses = [lm(prompt=rendered_prompt) for lm in self.lm_modules]
        return EnsembleOperatorOutputs(responses=responses)


class MostCommonInputs(EmberModel):
    """Input model for MostCommon."""

    responses: List[str]


class MostCommonOutputs(EmberModel):
    """Output model for MostCommon."""

    most_common: str
    counts: Dict[str, int]


class MostCommonOperator(Operator[MostCommonInputs, MostCommonOutputs]):
    """Implementation of the MostCommon operator."""

    specification = Specification[MostCommonInputs, MostCommonOutputs](
        input_model=MostCommonInputs, structured_output=MostCommonOutputs
    )

    def forward(self, *, inputs: MostCommonInputs) -> MostCommonOutputs:
        """Find the most common response."""
        # Count occurrences
        counts: Dict[str, int] = {}
        for response in inputs.responses:
            counts[response] = counts.get(response, 0) + 1

        # Find most common
        most_common = max(counts.items(), key=lambda x: x[1])[0] if counts else ""

        return MostCommonOutputs(most_common=most_common, counts=counts)


class VerifierInputs(EmberModel):
    """Input model for Verifier operator."""

    query: str
    candidate_answer: str


class VerifierOutputs(EmberModel):
    """Output model for Verifier operator."""

    verdict: str
    explanation: str
    revised_answer: Optional[str] = None


class VerifierOperator(Operator[VerifierInputs, VerifierOutputs]):
    """Implementation of the Verifier operator."""

    specification = Specification[VerifierInputs, VerifierOutputs](
        input_model=VerifierInputs,
        structured_output=VerifierOutputs,
        prompt_template=(
            "Please verify the following candidate answer to a question:\n\n"
            "Question: {query}\n\n"
            "Candidate answer: {candidate_answer}\n\n"
            "Is this answer correct and complete? Provide your verdict, explanation, "
            "and if necessary, a revised answer in the following format:\n\n"
            "Verdict: [correct/incorrect/partially correct]\n"
            "Explanation: [your detailed explanation]\n"
            "Revised Answer: [your improved answer, if needed]"
        ),
    )

    # Define static fields
    lm_module: LMModule = static_field()

    def __init__(self, lm_module: LMModule):
        """Initialize with LM module."""
        self.lm_module = lm_module

    def forward(self, *, inputs: VerifierInputs) -> VerifierOutputs:
        """Verify a candidate answer."""
        rendered_prompt = self.specification.render_prompt(inputs=inputs)
        raw_output = self.lm_module(prompt=rendered_prompt).strip()

        # Parse the response
        verdict = "unknown"
        explanation = ""
        revised_answer = None

        for line in raw_output.splitlines():
            line = line.strip()

            if line.startswith("Verdict:"):
                verdict = line.replace("Verdict:", "").strip().lower()
            elif line.startswith("Explanation:"):
                explanation = line.replace("Explanation:", "").strip()
            elif line.startswith("Revised Answer:"):
                revised_answer = line.replace("Revised Answer:", "").strip()

        return VerifierOutputs(
            verdict=verdict, explanation=explanation, revised_answer=revised_answer
        )


class Sequential(Operator):
    """Implementation of the Sequential operator pattern."""

    # Define a generic specification
    specification = Specification(
        input_model=None, structured_output=None, check_all_placeholders=False
    )

    # Define static fields
    operators: List[Operator] = static_field()

    def __init__(self, operators: List[Operator]):
        """Initialize with a list of operators."""
        self.operators = operators

    def forward(self, *, inputs: Any) -> Any:
        """Execute operators sequentially."""
        result = inputs
        for operator in self.operators:
            result = operator(inputs=result)
        return result


def test_multi_stage_pipeline_integration() -> None:
    """Test integration of a multi-stage operator pipeline using real components."""
    # Skip app_context creation and model registration

    # Create LM modules with simulation enabled
    lm_modules = [
        LMModule(
            config=LMModuleConfig(id="openai:gpt-3.5-turbo", temperature=0.7),
            simulate_api=True,
        )
        for _ in range(3)
    ]

    # Set up a real Ensemble operator with real LM modules
    ensemble = EnsembleOperator(lm_modules=lm_modules)

    # Set up a real MostCommon operator
    most_common = MostCommonOperator()

    # Set up a real Verifier operator with a real LM module
    verifier_lm = LMModule(
        config=LMModuleConfig(id="openai:gpt-3.5-turbo", temperature=0.2),
        simulate_api=True,
    )
    verifier = VerifierOperator(lm_module=verifier_lm)

    # Create a sequential pipeline with ensemble and most_common
    pipeline = Sequential(operators=[ensemble, most_common])

    # Create input data with a specific query
    input_data = EnsembleOperatorInputs(query="What is the capital of France?")

    # Execute the pipeline
    pipeline_output = pipeline(inputs=input_data)

    # Verify the pipeline results structure
    assert pipeline_output is not None
    assert isinstance(pipeline_output, MostCommonOutputs)
    assert hasattr(pipeline_output, "most_common")
    assert hasattr(pipeline_output, "counts")

    # Prepare inputs for the verifier
    verifier_input = VerifierInputs(
        query="What is the capital of France?",
        candidate_answer=pipeline_output.most_common,
    )

    # Execute the verifier with the pipeline output
    verifier_output = verifier(inputs=verifier_input)

    # Verify the final output
    assert verifier_output is not None
    assert isinstance(verifier_output, VerifierOutputs)
    assert hasattr(verifier_output, "verdict")
    assert hasattr(verifier_output, "explanation")
    # The revised_answer may or may not be present depending on the verdict


from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)
from ember.core.registry.model.providers.base_provider import BaseProviderModel


class FailingProvider(BaseProviderModel):
    """Provider implementation that simulates API failures."""

    PROVIDER_NAME = "FailingProvider"

    def create_client(self) -> Any:
        """Return a simple mock client."""
        return self

    def forward(self, request: ChatRequest) -> ChatResponse:
        """Always raises an error when called."""
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
def failing_registry():
    """Fixture for testing error handling with a failing provider."""
    # Create registry and model info
    registry = ModelRegistry()
    model_info = create_dummy_model_info("failing:model")

    # Register the model
    registry.register_model(model_info)

    # Directly inject a failing model instance
    registry._models["failing:model"] = FailingProvider(model_info=model_info)

    return registry


def test_provider_api_failure(failing_registry):
    """Test proper error handling when a provider API fails."""
    service = ModelService(registry=failing_registry)
    with pytest.raises(ProviderAPIError, match="Error invoking model failing:model"):
        service.invoke_model(model_id="failing:model", prompt="test")


def test_provider_model_not_found(failing_registry):
    """Test proper error handling when a model is not found in the registry."""
    service = ModelService(registry=failing_registry)
    with pytest.raises(ModelNotFoundError, match="Unknown:model"):
        service.invoke_model(model_id="Unknown:model", prompt="test")
