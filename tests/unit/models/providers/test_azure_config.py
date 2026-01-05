"""Unit tests for Azure shared configuration."""

from unittest.mock import patch

import pytest

from ember.models.providers.azure_openai.config import (
    AzureConfig,
    AzureRoute,
    canonicalize_model_id,
)


@pytest.fixture
def mock_context():
    with patch("ember._internal.context.runtime.EmberContext") as MockContext:
        ctx = MockContext.current.return_value
        # Default config
        ctx.get_config.side_effect = lambda key: {
            "providers.azure_openai.endpoint": "https://default.openai.azure.com",
            "providers.azure_openai.api_version": "2024-10-21",
            "providers.azure_openai.shards": {
                "gpt-4-turbo": {
                    "deployment": "gpt-4-turbo-deploy",
                    "endpoint": "https://shard1.openai.azure.com",
                    "model": "gpt-4-turbo",
                },
                "gpt-35-turbo": {
                    "endpoint": "https://shard2.openai.azure.com",
                    "use_responses_api": True,
                }
            }
        }.get(key)
        yield ctx


def test_azure_config_loading(mock_context):
    config = AzureConfig()

    assert len(config.routes) == 2

    # Check shard 1
    route1 = config.get_route("gpt-4-turbo")
    assert route1 is not None
    assert route1.deployment == "gpt-4-turbo-deploy"
    assert route1.endpoint == "https://shard1.openai.azure.com"
    assert route1.model == "gpt-4-turbo"
    assert route1.responses is False

    # Check shard 2
    route2 = config.get_route("gpt-35-turbo")
    assert route2 is not None
    assert route2.deployment == "gpt-35-turbo" # Default to key name
    assert route2.endpoint == "https://shard2.openai.azure.com"
    assert route2.responses is True


def test_canonicalize_model_id_with_mapping(mock_context):
    # Mock AzureConfig to be used by canonicalize_model_id
    with patch("ember.models.providers.azure_openai.config.AzureConfig") as MockConfig:
        config_instance = MockConfig.return_value
        config_instance.get_route.side_effect = lambda model_id: {
            "gpt-4-turbo": AzureRoute(
                deployment="gpt-4-turbo-deploy",
                endpoint="...",
                api_version="...",
                model="gpt-4-turbo",
            )
        }.get(model_id)

        # Test mapping
        assert canonicalize_model_id("azure_openai/gpt-4-turbo") == "gpt-4-turbo"
        assert canonicalize_model_id("gpt-4-turbo") == "gpt-4-turbo"

        # Test no mapping
        assert canonicalize_model_id("azure_openai/unknown-model") == "unknown-model"
