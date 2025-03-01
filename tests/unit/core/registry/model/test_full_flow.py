"""Integration tests for the full flow from configuration loading to model invocation.
Uses a dummy provider to simulate a full end-to-end scenario.
"""

import threading
from pathlib import Path
from typing import Any
from textwrap import dedent

import pytest

from ember.core.registry.model.config.settings import initialize_ember
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.base.services.usage_service import UsageService
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo


def create_dummy_config(tmp_path: Path) -> Path:
    """Creates a dummy config file for integration testing."""
    config_content = dedent(
        """
    registry:
      auto_register: true
      models:
        - id: "openai:gpt-4o"
          name: "Test Model"
          cost:
            input_cost_per_thousand: 1.0
            output_cost_per_thousand: 2.0
          rate_limit:
            tokens_per_minute: 1000
            requests_per_minute: 100
          provider:
            name: "DummyFactoryProvider"
            default_api_key: "dummy_key"
            base_url: "https://api.dummy.example"
          api_key: "dummy_key"
    """
    )
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

    # Patching the factory to use DummyProvider.
    monkeypatch.setattr(
        "ember.core.registry.model.base.registry.factory.ModelFactory.create_model_from_info",
        lambda *, model_info: DummyProvider(model_info),
    )

    # Also patch the registry's register_model method to ensure our model info is also registered.
    def mock_register_model(self, model_info: ModelInfo) -> None:
        """Mock registration that adds the model to both _models and _model_infos."""
        self._model_infos[model_info.id] = model_info
        self._models[model_info.id] = DummyProvider(model_info)

    from ember.core.registry.model.base.registry.model_registry import ModelRegistry

    monkeypatch.setattr(ModelRegistry, "register_model", mock_register_model)


def test_full_flow_concurrent_invocations(tmp_path, monkeypatch):
    """Ensure the registry handles concurrent invocations correctly."""
    config_path = create_dummy_config(tmp_path)
    registry = initialize_ember(
        config_path=str(config_path), auto_register=True, auto_discover=False
    )

    # Ensure model is registered
    if "openai:gpt-4o" not in registry.list_models():
        registry.register_model(create_dummy_model_info("openai:gpt-4o"))

    usage_service = UsageService()
    service = ModelService(registry=registry, usage_service=usage_service)

    def invoke_concurrently():
        resp = service.invoke_model(model_id="openai:gpt-4o", prompt="Concurrent test")
        # For demonstration purposes, we check a hypothetical substring in the response
        assert "Integrated: Concurrent test" in resp.data

    threads = [threading.Thread(target=invoke_concurrently) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def create_dummy_model_info(model_id: str) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        name="Test Model",
        cost=ModelCost(input_cost_per_thousand=1.0, output_cost_per_thousand=2.0),
        rate_limit=RateLimit(tokens_per_minute=1000, requests_per_minute=100),
        provider=ProviderInfo(
            name="DummyFactoryProvider",
            default_api_key="dummy_key",
            base_url="https://api.dummy.example",
        ),
        api_key="dummy_key",
    )
