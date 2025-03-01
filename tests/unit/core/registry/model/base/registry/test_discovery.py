"""Unit tests for the ModelDiscoveryService.
Tests caching, thread safety, merging of discovered models with local config, and error propagation.
"""

import threading
import time
import pytest
from typing import Any, Dict

from ember.core.registry.model.base.registry.discovery import ModelDiscoveryService
from ember.core.registry.model.providers.base_discovery import BaseDiscoveryProvider


class MockDiscoveryProvider(BaseDiscoveryProvider):
    """Mock provider for testing discovery service."""

    def fetch_models(self) -> Dict[str, Dict[str, Any]]:
        return {"mock:model": {"name": "Mock Model"}}


def test_discovery_service_fetch_and_cache() -> None:
    """Test that discovery service fetches models and then caches them."""
    service = ModelDiscoveryService(ttl=2)
    service.providers = [MockDiscoveryProvider()]
    models = service.discover_models()
    assert "mock:model" in models

    # Cache hit
    cached_models = service.discover_models()
    assert cached_models == models

    # Wait for cache expiration
    time.sleep(2.1)
    refreshed_models = service.discover_models()
    assert refreshed_models == models


def test_discovery_service_thread_safety() -> None:
    """Test thread safety of discovery service under concurrent access."""
    service = ModelDiscoveryService(ttl=3600)
    service.providers = [MockDiscoveryProvider()]

    def discover() -> None:
        models = service.discover_models()
        assert "mock:model" in models

    threads = [threading.Thread(target=discover) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_discovery_service_merge_with_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test merging discovered models with local configuration overrides."""
    from ember.core.registry.model.base.schemas.model_info import ModelInfo

    class MockEmberSettings:
        registry = type(
            "Registry",
            (),
            {
                "models": [
                    ModelInfo(
                        id="mock:model",
                        name="Mock Model Override",
                        cost={
                            "input_cost_per_thousand": 1.0,
                            "output_cost_per_thousand": 2.0,
                        },
                        rate_limit={
                            "tokens_per_minute": 1000,
                            "requests_per_minute": 100,
                        },
                        provider={
                            "name": "MockProvider",
                            "default_api_key": "mock_key",
                        },
                        api_key="mock_key",
                    )
                ]
            },
        )()

    monkeypatch.setattr(
        "ember.core.registry.model.config.settings.EmberSettings",
        MockEmberSettings,
    )

    service = ModelDiscoveryService()
    discovered = {"mock:model": {"id": "mock:model", "name": "Mock Model"}}
    merged = service.merge_with_config(discovered=discovered)
    assert "mock:model" in merged
    assert merged["mock:model"].name == "Mock Model Override"


def test_discovery_service_error_propagation() -> None:
    """Test that discovery service raises a ModelDiscoveryError when all providers fail."""
    from ember.core.registry.model.providers.base_discovery import (
        ModelDiscoveryError,
    )

    class FailingDiscoveryProvider(BaseDiscoveryProvider):
        def fetch_models(self) -> Dict[str, Dict[str, Any]]:
            raise Exception("Intentional failure for testing.")

    service = ModelDiscoveryService(ttl=3600)
    service.providers = [FailingDiscoveryProvider()]
    with pytest.raises(ModelDiscoveryError, match="No models discovered. Errors:"):
        service.discover_models()


@pytest.mark.asyncio
async def test_discovery_service_async_fetch_and_cache():
    """Test async discovery and caching."""
    service = ModelDiscoveryService(ttl=2)
    service.providers = [MockDiscoveryProvider()]
    models = await service.discover_models_async()
    assert "mock:model" in models
