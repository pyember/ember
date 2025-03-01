"""Unit tests for ProviderInfo schema.
Ensures that ProviderInfo initializes correctly and that optional fields default appropriately.
"""

import pytest

from ember.core.registry.model.base.schemas.provider_info import ProviderInfo


def test_provider_info_validation() -> None:
    """Test that ProviderInfo is correctly populated with given values."""
    provider = ProviderInfo(
        name="TestProvider",
        default_api_key="test_key",
        base_url="https://api.test.com",
        custom_args={"param": "value"},
    )
    assert provider.name == "TestProvider"
    assert provider.default_api_key == "test_key"
    assert provider.base_url == "https://api.test.com"
    assert provider.custom_args == {"param": "value"}


def test_provider_info_optional_fields() -> None:
    """Test ProviderInfo when optional fields are omitted."""
    provider = ProviderInfo(name="TestProvider")
    assert provider.default_api_key is None
    assert provider.base_url is None
    assert provider.custom_args == {}
