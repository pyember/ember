# """Integration tests for the model registry system.

# This module tests the complete flow from provider discovery to model registration and usage,
# verifying that all components work together correctly.
# """

# import os
# import pytest
# from typing import Dict, Any, List
# from unittest.mock import patch, MagicMock

# from ember.core.registry.model.base.registry.model_registry import ModelRegistry
# from ember.core.registry.model.base.schemas.model_info import ModelInfo, ProviderInfo
# from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
# from ember.core.registry.model.base.schemas.chat_schemas import (
#     ChatRequest,
#     ChatResponse,
# )
# from ember.core.registry.model.providers.base_provider import BaseProviderModel
# from ember.core.registry.model.providers.base_discovery import BaseDiscoveryProvider
# from ember.core.registry.model.base.registry.factory import ModelFactory


# class MockProvider(BaseProviderModel):
#     """Mock provider for integration tests."""

#     PROVIDER_NAME = "MockProvider"

#     def __init__(self, model_info: ModelInfo) -> None:
#         """Initialize the provider with model info."""
#         self.model_info = model_info
#         self.client = self.create_client()
#         self.calls: List[str] = []

#     def create_client(self) -> Any:
#         """Create a minimal mock client."""
#         return self

#     def forward(self, request: ChatRequest) -> ChatResponse:
#         """Process a request and return a mock response."""
#         self.calls.append(request.prompt)
#         return ChatResponse(data=f"Response to: {request.prompt}")


# class MockDiscoveryProvider(BaseDiscoveryProvider):
#     """Mock discovery provider for integration tests."""

#     def __init__(self, models: Dict[str, Dict[str, Any]]) -> None:
#         """Initialize with predefined models."""
#         self.models = models

#     def fetch_models(self) -> Dict[str, Dict[str, Any]]:
#         """Return the predefined models."""
#         return self.models


# def create_mock_model_info(
#     model_id: str, provider_name: str = "MockProvider"
# ) -> ModelInfo:
#     """Create a model info object for testing."""
#     return ModelInfo(
#         id=model_id,
#         name=f"Test {model_id}",
#         cost=ModelCost(input_cost_per_thousand=1.0, output_cost_per_thousand=2.0),
#         rate_limit=RateLimit(tokens_per_minute=1000, requests_per_minute=100),
#         provider=ProviderInfo(name=provider_name, default_api_key="test_key"),
#         api_key="test_key",
#     )


# @pytest.fixture
# def setup_environment():
#     """Setup test environment with required API keys."""
#     with patch.dict(
#         "os.environ",
#         {
#             "MOCK_API_KEY": "test_mock_key",
#         },
#     ):
#         yield


# @pytest.fixture
# def mock_discovery_service():
#     """Setup mock discovery service."""
#     # Create mock discovered models
#     discovered_models = {
#         "mock:model1": {"model_id": "mock:model1", "model_name": "Mock Model 1"},
#         "mock:model2": {"model_id": "mock:model2", "model_name": "Mock Model 2"},
#     }

#     # Create discovery provider
#     discovery_provider = MockDiscoveryProvider(discovered_models)

#     # Patch ModelDiscoveryService to use our mock provider
#     with patch(
#         "ember.core.registry.model.base.registry.discovery.ModelDiscoveryService"
#     ) as mock_service:
#         # Mock the service instance
#         service_instance = MagicMock()
#         service_instance.discover_models.return_value = discovered_models
#         service_instance.merge_with_config.return_value = {
#             model_id: create_mock_model_info(model_id)
#             for model_id in discovered_models.keys()
#         }
#         mock_service.return_value = service_instance

#         yield service_instance


# @pytest.fixture
# def mock_model_factory():
#     """Setup mock model factory."""
#     # Register our mock provider
#     ModelFactory._provider_cache = {"MockProvider": MockProvider}

#     yield

#     # Cleanup
#     ModelFactory._provider_cache = None


# def test_end_to_end_registry_workflow(
#     setup_environment, mock_discovery_service, mock_model_factory
# ):
#     """Test complete end-to-end workflow from discovery to model usage."""
#     # Create registry
#     registry = ModelRegistry()

#     # Discover models
#     discovered_models = registry.discover_models()

#     # Verify models were discovered
#     assert len(discovered_models) == 2
#     assert "mock:model1" in discovered_models
#     assert "mock:model2" in discovered_models

#     # Verify mock discovery service was called properly
#     mock_discovery_service.discover_models.assert_called_once()
#     mock_discovery_service.merge_with_config.assert_called_once()

#     # Check registry state
#     assert "mock:model1" in registry.list_models()
#     assert "mock:model2" in registry.list_models()

#     # Get and use a model
#     model = registry.get_model("mock:model1")
#     response = model("Hello, test")

#     # Verify model was used
#     assert isinstance(model, MockProvider)
#     assert "Hello, test" in model.calls
#     assert response.data == "Response to: Hello, test"

#     # Get model info
#     model_info = registry.get_model_info("mock:model1")
#     assert model_info is not None
#     assert model_info.id == "mock:model1"
#     assert model_info.provider.name == "MockProvider"


# def test_registry_with_explicit_registration(mock_model_factory):
#     """Test registry with explicit model registration rather than discovery."""
#     # Create registry
#     registry = ModelRegistry()

#     # Create model info
#     model_info = create_mock_model_info("explicit:model")

#     # Register directly
#     registry.register_model(model_info)

#     # Verify registration
#     assert "explicit:model" in registry.list_models()

#     # Get and use the model
#     model = registry.get_model("explicit:model")
#     response = model("Explicit registration")

#     # Verify model behavior
#     assert isinstance(model, MockProvider)
#     assert "Explicit registration" in model.calls
#     assert response.data == "Response to: Explicit registration"


# def test_registry_update_model(mock_model_factory):
#     """Test updating an existing model in the registry."""
#     # Create registry
#     registry = ModelRegistry()

#     # Create initial model info
#     initial_info = create_mock_model_info("update:model")
#     initial_info.name = "Initial Name"

#     # Register the model
#     registry.register_model(initial_info)

#     # Create updated model info with same ID but different details
#     updated_info = create_mock_model_info("update:model")
#     updated_info.name = "Updated Name"

#     # Update the model
#     registry.register_or_update_model(updated_info)

#     # Get the model info and verify it was updated
#     model_info = registry.get_model_info("update:model")
#     assert model_info is not None
#     assert model_info.name == "Updated Name"

#     # Verify model instance was created
#     model = registry.get_model("update:model")
#     assert isinstance(model, MockProvider)


# def test_unregister_and_reregister(mock_model_factory):
#     """Test unregistering and then re-registering a model."""
#     # Create registry
#     registry = ModelRegistry()

#     # Create model info
#     model_info = create_mock_model_info("temp:model")

#     # Register, unregister, then register again
#     registry.register_model(model_info)
#     assert "temp:model" in registry.list_models()

#     registry.unregister_model("temp:model")
#     assert "temp:model" not in registry.list_models()

#     # Re-register
#     registry.register_model(model_info)
#     assert "temp:model" in registry.list_models()

#     # Model should work after re-registration
#     model = registry.get_model("temp:model")
#     response = model("After re-registration")
#     assert response.data == "Response to: After re-registration"


# def test_registry_error_handling(mock_model_factory):
#     """Test registry error handling with invalid models."""
#     # Create registry
#     registry = ModelRegistry()

#     # Try to get a non-existent model
#     with pytest.raises(Exception) as exc_info:
#         registry.get_model("nonexistent:model")

#     assert "not found" in str(exc_info.value).lower()

#     # Create a model with invalid provider
#     invalid_info = create_mock_model_info(
#         "invalid:provider", provider_name="NonExistentProvider"
#     )

#     # Registering should work (it doesn't validate provider at registration time)
#     registry.register_model(invalid_info)

#     # But getting the model should fail when it tries to instantiate it
#     with pytest.raises(Exception) as exc_info:
#         registry.get_model("invalid:provider")

#     assert "provider" in str(exc_info.value).lower()
