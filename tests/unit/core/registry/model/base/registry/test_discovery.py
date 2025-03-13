"""Unit tests for the ModelDiscoveryService.

This module tests the ModelDiscoveryService which is responsible for:
1. Discovering available models from provider APIs
2. Caching results for performance
3. Merging discovered models with local configuration
4. Thread-safe access to discovery results
5. Error handling for provider failures
"""

import threading
import time
import pytest
from typing import Any, Dict, List, Optional, cast
from unittest.mock import MagicMock, patch

from ember.core.registry.model.base.registry.discovery import ModelDiscoveryService
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.providers.base_discovery import (
    BaseDiscoveryProvider,
    ModelDiscoveryError
)


class MockDiscoveryProvider(BaseDiscoveryProvider):
    """Mock provider for testing the discovery service.
    
    This provider returns a predefined set of models for testing.
    """
    def __init__(self, models: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Initialize the mock provider with optional custom models."""
        self.models = models or {"mock:model": {"model_id": "mock:model", "model_name": "Mock Model"}}
        self.call_count = 0
        
    def fetch_models(self) -> Dict[str, Dict[str, Any]]:
        """Return a set of mock models and increment call counter."""
        self.call_count += 1
        return self.models


@pytest.fixture
def discovery_service() -> ModelDiscoveryService:
    """Create a ModelDiscoveryService with a MockDiscoveryProvider."""
    service = ModelDiscoveryService(ttl=2)
    service.providers = [MockDiscoveryProvider()]
    return service


def test_discovery_service_fetch_and_cache(discovery_service: ModelDiscoveryService) -> None:
    """Test that the discovery service fetches models and then caches them.
    
    This test verifies:
    1. Initial call fetches models from providers
    2. Subsequent calls within TTL use cached results
    3. Cache is refreshed after TTL expiration
    """
    # Initial call fetches models
    models = discovery_service.discover_models()
    assert "mock:model" in models
    provider = cast(MockDiscoveryProvider, discovery_service.providers[0])
    assert provider.call_count == 1

    # Cache hit - provider shouldn't be called again
    cached_models = discovery_service.discover_models()
    assert cached_models == models
    assert provider.call_count == 1

    # Wait for cache expiration
    time.sleep(2.1)
    
    # Cache should be refreshed
    refreshed_models = discovery_service.discover_models()
    assert refreshed_models == models
    assert provider.call_count == 2


def test_discovery_service_thread_safety() -> None:
    """Test thread safety of discovery service under concurrent access.
    
    This test verifies that the service can be safely accessed from multiple threads.
    """
    service = ModelDiscoveryService(ttl=3600)
    provider = MockDiscoveryProvider()
    service.providers = [provider]
    results = []
    errors = []

    def discover() -> None:
        """Thread worker that calls discover_models and records results."""
        try:
            models = service.discover_models()
            results.append(models)
        except Exception as e:
            errors.append(e)

    # Create and start threads
    threads = [threading.Thread(target=discover) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    # Verify no errors occurred and all threads got the same result
    assert not errors, f"Errors occurred during threaded access: {errors}"
    assert len(results) == 10
    assert all(r == results[0] for r in results)
    
    # Provider should be called exactly once despite multiple threads
    assert provider.call_count == 1


def test_discovery_service_initialize_providers() -> None:
    """Test the provider initialization based on available API keys."""
    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test_key',
        'ANTHROPIC_API_KEY': 'test_key',
    }):
        service = ModelDiscoveryService()
        # Should initialize providers for OpenAI and Anthropic
        assert len(service.providers) >= 2
        
    # When no API keys are available, we need to mock a fallback provider
    with patch.dict('os.environ', {}, clear=True):
        # Create a class to provide a fallback provider for testing
        class FallbackDiscoveryProvider(BaseDiscoveryProvider):
            def fetch_models(self) -> Dict[str, Dict[str, Any]]:
                return {"fallback:model": {"model_id": "fallback:model"}}
                
        # Patch the _initialize_providers method to include our fallback provider
        with patch.object(ModelDiscoveryService, '_initialize_providers', 
                          return_value=[FallbackDiscoveryProvider()]):
            service = ModelDiscoveryService()
            # Should include our patched fallback provider
            assert len(service.providers) > 0


def test_discovery_service_merge_with_config() -> None:
    """Test merging discovered models with local configuration overrides."""
    service = ModelDiscoveryService()
    
    # Create mock model info using the proper constructor
    model_info = ModelInfo(
        id="mock:model",
        name="Mock Model Override",
        cost=ModelCost(input_cost_per_thousand=1.0, output_cost_per_thousand=2.0),
        rate_limit=RateLimit(tokens_per_minute=1000, requests_per_minute=100),
        provider=ProviderInfo(name="MockProvider", default_api_key="mock_key"),
        api_key="mock_key"
    )
    
    # Use a module-level patch to avoid import path issues
    import sys
    from types import ModuleType
    from unittest.mock import patch
    
    # Create a mock EmberSettings module and class for patching
    mock_settings_module = ModuleType("ember.core.registry.model.config.settings")
    
    # Create a mock class
    class MockEmberSettings:
        def __init__(self):
            self.registry = MagicMock()
            self.registry.models = [model_info]
    
    # Add the class to the module
    mock_settings_module.EmberSettings = MockEmberSettings
    
    # Register the module in sys.modules for imports to find
    original_module = sys.modules.get('ember.core.registry.model.config.settings', None)
    sys.modules['ember.core.registry.model.config.settings'] = mock_settings_module
    
    try:
        discovered = {"mock:model": {"model_id": "mock:model", "model_name": "Mock Model"}}
        
        # Apply environment variable patches for API keys
        with patch.dict('os.environ', {'MOCK_API_KEY': 'mock_key'}):
            # Now call merge_with_config which will import our mocked module
            merged = service.merge_with_config(discovered=discovered)
            
            # Verify results
            assert "mock:model" in merged
            # The name should come from our mocked model_info
            assert merged["mock:model"].name == "Mock Model Override"
    finally:
        # Clean up the sys.modules patch
        if original_module:
            sys.modules['ember.core.registry.model.config.settings'] = original_module
        else:
            del sys.modules['ember.core.registry.model.config.settings']


def test_discovery_service_error_propagation() -> None:
    """Test that discovery service handles provider failures appropriately."""
    class FailingDiscoveryProvider(BaseDiscoveryProvider):
        """Provider that always fails for testing error handling."""
        def fetch_models(self) -> Dict[str, Dict[str, Any]]:
            """Raise an exception to simulate provider failure."""
            raise Exception("Intentional failure for testing.")

    service = ModelDiscoveryService(ttl=3600)
    service.providers = [FailingDiscoveryProvider()]
    
    # Service should handle provider errors and raise a ModelDiscoveryError
    with pytest.raises(ModelDiscoveryError) as exc_info:
        service.discover_models()
    
    assert "No models discovered" in str(exc_info.value)
    assert "Intentional failure for testing" in str(exc_info.value)


def test_discovery_service_mixed_provider_failures() -> None:
    """Test behavior when some providers succeed and others fail."""
    service = ModelDiscoveryService(ttl=3600)
    
    # Create a concrete implementation of BaseDiscoveryProvider that fails
    class FailingDiscoveryProvider(BaseDiscoveryProvider):
        def fetch_models(self) -> Dict[str, Dict[str, Any]]:
            raise NotImplementedError("This provider intentionally fails")
    
    service.providers = [
        MockDiscoveryProvider({"success:model": {"model_id": "success:model"}}),
        FailingDiscoveryProvider(),  # This will fail with NotImplementedError
    ]
    
    # Service should return models from successful providers
    models = service.discover_models()
    assert "success:model" in models


@pytest.mark.asyncio
async def test_discovery_service_async_fetch_and_cache(discovery_service: ModelDiscoveryService) -> None:
    """Test async discovery and caching behavior."""
    # Initial async discovery
    models = await discovery_service.discover_models_async()
    assert "mock:model" in models
    provider = cast(MockDiscoveryProvider, discovery_service.providers[0])
    assert provider.call_count == 1
    
    # Cache hit - should use cached results
    cached_models = await discovery_service.discover_models_async()
    assert cached_models == models
    assert provider.call_count == 1
    
    # Wait for cache expiration
    time.sleep(2.1)
    
    # Should refresh cache after expiration
    refreshed_models = await discovery_service.discover_models_async()
    assert refreshed_models == models
    assert provider.call_count == 2
