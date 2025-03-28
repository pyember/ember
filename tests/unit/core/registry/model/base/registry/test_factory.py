"""Unit tests for the ModelFactory.

Tests provider discovery, instantiation, and error handling in the model factory system.
"""

import sys
from types import ModuleType
from typing import Any, Dict, Type
from unittest.mock import MagicMock, patch

import pytest

from ember.core.registry.model.base.registry.factory import (
    ModelFactory,
    discover_providers_in_package,
)
from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.base.schemas.model_info import ModelInfo
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    ProviderConfigError,
)
from ember.core.registry.model.providers.base_provider import BaseProviderModel


# Dummy provider classes for testing
class DummyProviderModel(BaseProviderModel):
    """Test provider for factory tests."""

    PROVIDER_NAME = "DummyProvider"

    def __init__(self, model_info: ModelInfo) -> None:
        self.model_info = model_info

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return f"Echo: {prompt}"

    def create_client(self) -> None:
        return None

    def forward(self, request):
        return None


class AnotherProviderModel(BaseProviderModel):
    """Second test provider for testing multiple provider registration."""

    PROVIDER_NAME = "AnotherProvider"

    def __init__(self, model_info: ModelInfo) -> None:
        self.model_info = model_info

    def create_client(self) -> None:
        return None

    def forward(self, request):
        return None


class InvalidProviderModel:
    """Invalid provider class that doesn't inherit from BaseProviderModel."""

    PROVIDER_NAME = "InvalidProvider"


# Dummy discover function that returns our dummy provider.
def dummy_discover_providers(
    *, package_name: str, package_path: str
) -> Dict[str, Type[BaseProviderModel]]:
    """Mock discover function that returns our test providers."""
    return {"DummyProvider": DummyProviderModel}


@pytest.fixture(autouse=True)
def patch_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the factory for testing purposes.

    This function directly patches the ModelFactory implementation since
    monkeypatching the import path has issues with duplicate modules.
    """
    # Import factory module directly
    from ember.core.registry.model.base.registry import factory

    # Apply patch directly to the module
    monkeypatch.setattr(
        factory, "discover_providers_in_package", dummy_discover_providers
    )

    # Also reset the cached providers
    from ember.core.registry.model.base.registry.factory import ModelFactory

    ModelFactory._provider_cache = None
    ModelFactory._testing_mode = False


def create_dummy_model_info(model_id: str = "openai:gpt-4o") -> ModelInfo:
    """Create a ModelInfo instance for testing the factory."""
    return ModelInfo(
        id=model_id,
        name="DummyFactoryModel",
        cost=ModelCost(input_cost_per_thousand=1.0, output_cost_per_thousand=2.0),
        rate_limit=RateLimit(tokens_per_minute=1000, requests_per_minute=100),
        provider=ProviderInfo(name="DummyProvider", default_api_key="dummy_key"),
        api_key="dummy_key",
    )


@patch.object(
    ModelFactory, "_get_providers", return_value={"DummyProvider": DummyProviderModel}
)
def test_create_model_from_info_success(mock_get_providers) -> None:
    """Test that ModelFactory.create_model_from_info successfully creates a DummyProviderModel."""
    dummy_info = create_dummy_model_info("openai:gpt-4o")
    model_instance = ModelFactory.create_model_from_info(model_info=dummy_info)
    assert isinstance(model_instance, DummyProviderModel)
    assert model_instance.model_info.id == "openai:gpt-4o"


def test_create_model_from_info_invalid() -> None:
    """Test that an invalid model id causes ProviderConfigError."""
    dummy_info = create_dummy_model_info("invalid:model")

    # Use patch to avoid import path issues
    from ember.core.registry.model.base.registry import factory
    from ember.core.registry.model.config import model_enum

    # Keep original function for restoration after test
    original_parse_func = model_enum.parse_model_str

    try:
        # Directly modify the function
        def mock_parse_model_str(model_str: str) -> str:
            raise ValueError("Invalid model ID format")

        # Apply the patch directly to the imported module
        model_enum.parse_model_str = mock_parse_model_str

        # The factory imports parse_model_str directly, so we need to patch that reference too
        factory.parse_model_str = mock_parse_model_str

        # Now test the factory function - should raise ProviderConfigError
        with pytest.raises(ProviderConfigError):
            ModelFactory.create_model_from_info(model_info=dummy_info)
    finally:
        # Restore the original function
        model_enum.parse_model_str = original_parse_func
        factory.parse_model_str = original_parse_func


def test_unknown_provider_error() -> None:
    """Test error handling when an unknown provider is specified."""
    # Create model info with non-existent provider
    dummy_info = create_dummy_model_info("openai:gpt-4o")
    dummy_info.provider.name = "NonExistentProvider"

    # Ensure the factory is initialized with only our test provider
    with patch.object(
        ModelFactory,
        "_get_providers",
        return_value={"DummyProvider": DummyProviderModel},
    ):
        # Should raise ProviderConfigError with available providers list
        with pytest.raises(ProviderConfigError) as exc_info:
            ModelFactory.create_model_from_info(model_info=dummy_info)

        # Error should include available providers in the message
        assert "Unsupported provider" in str(exc_info.value)
        assert "DummyProvider" in str(exc_info.value)


def test_register_custom_provider() -> None:
    """Test registration and usage of a custom provider."""
    # First ensure cache is cleared
    ModelFactory._provider_cache = None

    # Register our custom provider
    ModelFactory.register_custom_provider(
        provider_name="CustomTestProvider", provider_class=DummyProviderModel
    )

    # Verify it was registered
    providers = ModelFactory._get_providers()
    assert "CustomTestProvider" in providers

    # Create model info for the custom provider
    custom_info = create_dummy_model_info("custom:model")
    custom_info.provider.name = "CustomTestProvider"

    # Create model using the custom provider
    model = ModelFactory.create_model_from_info(model_info=custom_info)

    # Verify correct model was created
    assert isinstance(model, DummyProviderModel)
    assert model.model_info.id == "custom:model"


def test_register_invalid_provider() -> None:
    """Test that registering an invalid provider class raises ValueError."""
    # Should raise error when non-BaseProviderModel class is registered
    with pytest.raises(ValueError) as exc_info:
        ModelFactory.register_custom_provider(
            provider_name="InvalidProvider",
            provider_class=InvalidProviderModel,  # Does not inherit from BaseProviderModel
        )

    assert "must subclass BaseProviderModel" in str(exc_info.value)


def test_testing_mode() -> None:
    """Test that testing mode disables dynamic provider discovery."""
    # Reset provider cache
    ModelFactory._provider_cache = None

    # First test with normal mode
    ModelFactory.disable_testing_mode()

    # In normal mode, _get_providers should try to discover dynamic providers
    with patch("importlib.import_module") as mock_import:
        # Setup the mock to simulate package discovery
        mock_module = MagicMock()
        mock_module.__name__ = "ember.core.registry.model.providers"
        mock_module.__file__ = "/path/to/providers"
        mock_import.return_value = mock_module

        # Execute _get_providers
        providers = ModelFactory._get_providers()

        # Should have tried to import for discovery
        mock_import.assert_called_once_with("ember.core.registry.model.providers")

    # Now test with testing mode enabled
    ModelFactory.enable_testing_mode()

    # Reset cache to force re-initialization
    ModelFactory._provider_cache = None

    # In testing mode, _get_providers should skip dynamic discovery
    with patch("importlib.import_module") as mock_import:
        providers = ModelFactory._get_providers()

        # Should not try to import
        mock_import.assert_not_called()


def test_discover_providers_in_package() -> None:
    """Test the discover_providers_in_package function directly."""
    # Create a temporary mock module structure
    test_module = ModuleType("test_module")

    # Add our provider classes to the module
    test_module.DummyProviderModel = DummyProviderModel
    test_module.AnotherProviderModel = AnotherProviderModel
    test_module.InvalidProviderModel = InvalidProviderModel
    test_module.SomeOtherClass = object  # Non-provider class

    # Register the module
    original_module = sys.modules.get("test_module", None)
    sys.modules["test_module"] = test_module

    try:
        # Mock package structure
        with patch("pkgutil.walk_packages") as mock_walk:
            mock_walk.return_value = [
                (None, "test_module.provider1", False),  # Should be included
                (None, "test_module.provider2", False),  # Should be included
                (
                    None,
                    "test_module.subpackage",
                    True,
                ),  # Should be skipped (is a package)
                (
                    None,
                    "test_module.base_discovery",
                    False,
                ),  # Should be skipped (name blacklist)
            ]

            # Mock importlib.import_module to return our test module
            def mock_import(name):
                if name.startswith("test_module"):
                    return test_module
                raise ImportError(f"Cannot import {name}")

            with patch("importlib.import_module", side_effect=mock_import):
                # Run discovery
                discovered = discover_providers_in_package(
                    package_name="test_module", package_path="/path/to/module"
                )

                # Should find our provider classes
                assert len(discovered) == 2
                assert "DummyProvider" in discovered
                assert "AnotherProvider" in discovered
                assert discovered["DummyProvider"] is DummyProviderModel
                assert discovered["AnotherProvider"] is AnotherProviderModel

                # Should not include invalid providers or non-provider classes
                assert "InvalidProvider" not in discovered
                assert "SomeOtherClass" not in discovered
    finally:
        # Clean up
        if original_module:
            sys.modules["test_module"] = original_module
        else:
            del sys.modules["test_module"]


def test_discover_provider_import_error() -> None:
    """Test handling of import errors during provider discovery."""
    # Mock a failing import during package walk
    with patch("pkgutil.walk_packages") as mock_walk:
        mock_walk.return_value = [
            (None, "test_module.failing_module", False),
        ]

        # Mock importlib.import_module to raise ImportError
        with patch(
            "importlib.import_module", side_effect=ImportError("Cannot import module")
        ):
            with patch("logging.Logger.warning") as mock_log:
                # Run discovery, should not raise but should log warning
                discovered = discover_providers_in_package(
                    package_name="test_module", package_path="/path/to/module"
                )

                # Should return empty dict but not fail
                assert discovered == {}

                # Should log warning
                mock_log.assert_called_once()
                assert "Failed to load provider module" in mock_log.call_args[0][0]


def test_provider_without_name() -> None:
    """Test handling of provider classes without PROVIDER_NAME attribute."""

    # Create a provider class without PROVIDER_NAME
    class NoNameProvider(BaseProviderModel):
        def create_client(self):
            return None

        def forward(self, request):
            return None

    # Create a temporary module with our provider
    test_module = ModuleType("test_module_noname")
    test_module.NoNameProvider = NoNameProvider

    # Register the module
    sys.modules["test_module_noname"] = test_module

    try:
        # Mock package walk to include our module
        with patch("pkgutil.walk_packages") as mock_walk:
            mock_walk.return_value = [
                (None, "test_module_noname.provider", False),
            ]

            # Mock import to return our module
            with patch("importlib.import_module", return_value=test_module):
                # Run discovery
                discovered = discover_providers_in_package(
                    package_name="test_module_noname", package_path="/path/to/module"
                )

                # Should find no valid providers since our class has no PROVIDER_NAME
                assert discovered == {}
    finally:
        # Clean up
        del sys.modules["test_module_noname"]
