"""Integration tests for the full flow from configuration loading to model invocation.
Uses a dummy provider to simulate a full end-to-end scenario.
"""

import threading
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from ember.core.config.manager import create_config_manager
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.base.services.usage_service import UsageService
from ember.core.registry.model.initialization import initialize_registry


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
def patch_factory() -> None:
    """Patch ModelFactory to always return a dummy provider for integration testing."""

    # Define a DummyProvider class for our tests
    class DummyProvider:
        def __init__(self, model_info: Any) -> None:
            self.model_info = model_info

        def __call__(self, prompt: str, **kwargs) -> Any:
            class DummyResponse:
                data = f"Integrated: {prompt}"
                usage = None

            return DummyResponse()

    # Import the modules directly rather than using string-based monkeypatching
    from ember.core.registry.model.base.registry import factory
    from ember.core.registry.model.base.registry.model_registry import ModelRegistry

    # Save original functions
    original_create_model = None
    original_register_model = None

    if hasattr(factory.ModelFactory, "create_model_from_info"):
        original_create_model = factory.ModelFactory.create_model_from_info

    if hasattr(ModelRegistry, "register_model"):
        original_register_model = ModelRegistry.register_model

    try:
        # Define a replacement for create_model_from_info
        def dummy_create_model_from_info(*, model_info):
            return DummyProvider(model_info)

        # Define a replacement for register_model
        def mock_register_model(self, model_info: ModelInfo) -> None:
            """Mock registration that adds the model to both _models and _model_infos."""
            self._model_infos[model_info.id] = model_info
            self._models[model_info.id] = DummyProvider(model_info)

        # Apply the patches directly
        factory.ModelFactory.create_model_from_info = dummy_create_model_from_info
        ModelRegistry.register_model = mock_register_model

        yield
    finally:
        # Restore original functions
        if original_create_model:
            factory.ModelFactory.create_model_from_info = original_create_model
        if original_register_model:
            ModelRegistry.register_model = original_register_model


def test_full_flow_concurrent_invocations(tmp_path, monkeypatch):
    """Ensure the registry handles concurrent invocations correctly."""
    config_path = create_dummy_config(tmp_path)
    config_manager = create_config_manager(config_path=str(config_path))
    registry = initialize_registry(config_manager=config_manager, auto_discover=False)

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
