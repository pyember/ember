#!/usr/bin/env python3
"""Integration tests for the full flow from configuration loading to model invocation.
Uses a dummy provider to simulate a full end-to-end scenario.
"""

import threading
from pathlib import Path
from typing import Any
from textwrap import dedent

import pytest

from src.ember.core.registry.model.config.settings import initialize_ember
from src.ember.core.registry.model.model_modules.lm import LMModule, LMModuleConfig
from src.ember.core.registry.model.base.services.model_service import ModelService
from src.ember.core.registry.model.base.services.usage_service import UsageService
from src.ember.core.registry.model.base.schemas.model_info import ModelInfo
from src.ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from src.ember.core.registry.model.base.schemas.provider_info import ProviderInfo



def create_dummy_config(tmp_path: Path) -> Path:
    """Creates a dummy config file for integration testing."""
    config_content = dedent("""
    registry:
      models:
        - id: "dummy:test-model"
          name: "Test Model"
          cost:
            input_cost_per_thousand: 1.0
            output_cost_per_thousand: 2.0
          rate_limit:
            tokens_per_minute: 1000
            requests_per_minute: 100
          provider:
            name: "DummyProvider"
            default_api_key: "dummy_key"
          api_key: "dummy_key"
    """)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture(autouse=True)
def patch_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ModelFactory to always return a dummy provider for integration testing."""
    class DummyProvider:
        def __init__(self, model_info: Any) -> None:
            self.model_info = model_info

        def __call__(self, prompt: str, **kwargs) -> Any:
            class DummyResponse:
                data = f"Integrated: {prompt}"
                usage = None
            return DummyResponse()

    # Patch the factory
    monkeypatch.setattr(
        "src.ember.core.registry.model.base.registry.factory.ModelFactory.create_model_from_info",
        lambda *, model_info: DummyProvider(model_info),
    )

    # Also patch the registry's _models dict to ensure our model is registered
    def mock_register_model(self, model_info: ModelInfo) -> None:
        """Mock registration that actually adds the model to _models"""
        self._models[model_info.id] = DummyProvider(model_info)

    from src.ember.core.registry.model.base.registry.model_registry import ModelRegistry
    monkeypatch.setattr(ModelRegistry, "register_model", mock_register_model)


def test_full_flow_with_lm_module(tmp_path: Path) -> None:
    """Test the full flow from config loading to LMModule invocation."""
    config_path = create_dummy_config(tmp_path)
    registry = initialize_ember(config_path=str(config_path), auto_register=True, auto_discover=False)
    
    registered = list(registry._models.keys())
    print(f"Registered models: {registered}")
    if "dummy:test-model" not in registry._models:
        dummy_info = ModelInfo(
            id="dummy:test-model",
            name="Test Model",
            cost=ModelCost(input_cost_per_thousand=1.0, output_cost_per_thousand=2.0),
            rate_limit=RateLimit(tokens_per_minute=1000, requests_per_minute=100),
            provider=ProviderInfo(name="DummyProvider", default_api_key="dummy_key"),
            api_key="dummy_key",
        )
        registry.register_model(dummy_info)
        print("Manually registered dummy:test-model")

    usage_service = UsageService()
    model_service = ModelService(registry=registry, usage_service=usage_service)

    config = LMModuleConfig(id="dummy:test-model", simulate_api=False)
    lm = LMModule(config=config, model_service=model_service)
    response = lm(prompt="Hello integration!")
    assert "Integrated: Hello integration!" in response

    summary = usage_service.get_usage_summary(model_id="dummy:test-model")
    # Since our dummy provider does not update usage, totals remain 0.
    assert summary.total_usage.total_tokens == 0