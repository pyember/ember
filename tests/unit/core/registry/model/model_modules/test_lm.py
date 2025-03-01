"""
Unit tests for LMModule in the LM (language model) module.
These tests check that LMModule:
  - returns a simulated response when simulate_api is True.
  - properly calls the underlying ModelService when simulate_api is False.
"""

import logging
from typing import Any

import pytest

from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig


class DummyModelService:
    """A dummy ModelService that returns a fixed ChatResponse."""

    def invoke_model(self, model_id: str, prompt: str, **kwargs: Any) -> Any:
        class DummyResponse:
            data = f"Dummy response for prompt: {prompt}"
            usage = None

        return DummyResponse()


@pytest.fixture
def dummy_service() -> DummyModelService:
    return DummyModelService()


def test_lm_module_simulate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LMModule returns a simulated response when simulate_api is True."""
    config = LMModuleConfig(id="openai:gpt-4o", temperature=0.8)
    # simulate_api=True should bypass model_service.invoke_model
    lm = LMModule(config=config, simulate_api=True)
    prompt = "Simulated test prompt"
    result = lm(prompt)
    expected = f"SIMULATED_RESPONSE: {prompt}"
    assert result == expected, "Simulated response does not match expected output."


def test_lm_module_forward_calls_service(dummy_service: DummyModelService) -> None:
    """Test that LMModule.forward calls the underlying model_service when simulate_api is False."""
    config = LMModuleConfig(
        id="openai:gpt-4o", temperature=0.8, cot_prompt="Think step-by-step"
    )
    # When simulate_api is False, LMModule should call the model service.
    lm = LMModule(config=config, model_service=dummy_service, simulate_api=False)
    prompt = "Test prompt"
    response = lm.forward(prompt)
    expected_fragment = "Dummy response for prompt:"
    # Since lm.forward returns a string, we check the content of the string directly.
    assert (
        expected_fragment in response
    ), "LMModule did not call the dummy service correctly."


def test_lm_module_assemble_full_prompt() -> None:
    """Test the private method _assemble_full_prompt for proper prompt merging."""
    config = LMModuleConfig(
        id="openai:gpt-4o",
        temperature=0.8,
        cot_prompt="Chain of Thought",
        persona="Friendly",
    )
    lm = LMModule(config=config, simulate_api=True)
    # Access the private method directly (acceptable in tests)
    full_prompt = lm._assemble_full_prompt("What is AI?")
    # Expected prompt: persona prepended, then the question, then chain-of-thought appended.
    assert "[Persona: Friendly]" in full_prompt
    assert "What is AI?" in full_prompt
    assert "Chain of Thought" in full_prompt
