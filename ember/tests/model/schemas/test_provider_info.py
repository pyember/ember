import pytest
from src.ember.registry.model.schemas.provider_info import ProviderInfo


def test_provider_info_basic():
    pi = ProviderInfo(name="Anthropic")
    assert pi.name == "Anthropic"
    assert pi.default_api_key is None
    assert pi.base_url is None
    assert pi.custom_args == {}


def test_provider_info_full():
    pi = ProviderInfo(
        name="Google",
        default_api_key="some_api_key",
        base_url="http://custom-endpoint",
        custom_args={"arg1": "val1"},
    )
    assert pi.name == "Google"
    assert pi.default_api_key == "some_api_key"
    assert pi.base_url == "http://custom-endpoint"
    assert pi.custom_args == {"arg1": "val1"}
