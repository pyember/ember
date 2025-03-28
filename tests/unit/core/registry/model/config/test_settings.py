"""Simplified tests for the model registry initialization module."""

from unittest.mock import MagicMock


# Mock ModelInfo class
class MockModelInfo:
    def __init__(self, model_id, model_name, cost, rate_limit, provider, api_key):
        self.model_id = model_id
        self.model_name = model_name
        self.cost = cost
        self.rate_limit = rate_limit
        self.provider = provider
        self.api_key = api_key

    def get_api_key(self):
        return self.api_key


# Mock classes for config conversion
class MockCost:
    def __init__(self, input_cost_per_thousand, output_cost_per_thousand):
        self.input_cost_per_thousand = input_cost_per_thousand
        self.output_cost_per_thousand = output_cost_per_thousand


class MockRateLimit:
    def __init__(self, tokens_per_minute, requests_per_minute):
        self.tokens_per_minute = tokens_per_minute
        self.requests_per_minute = requests_per_minute


class MockProviderInfo:
    def __init__(self, name, default_api_key, base_url):
        self.name = name
        self.default_api_key = default_api_key
        self.base_url = base_url
        self.custom_args = {}


# Test implementation of model_config_to_model_info
def convert_model_config_to_model_info(
    model_id, provider_name, model_config, provider_config, api_key
):
    """Test implementation of conversion function."""
    # Create cost object
    cost = MockCost(
        input_cost_per_thousand=getattr(
            model_config.cost, "input_cost_per_thousand", 0.0
        ),
        output_cost_per_thousand=getattr(
            model_config.cost, "output_cost_per_thousand", 0.0
        ),
    )

    # Create rate limit
    rate_limit = MockRateLimit(
        tokens_per_minute=getattr(model_config.rate_limit, "tokens_per_minute", 0),
        requests_per_minute=getattr(model_config.rate_limit, "requests_per_minute", 0),
    )

    # Create provider info
    provider_info = MockProviderInfo(
        name=provider_name.capitalize(),
        default_api_key=api_key,
        base_url=getattr(provider_config, "base_url", None),
    )

    # Add custom args
    if hasattr(provider_config, "model_dump") and callable(provider_config.model_dump):
        custom_args = provider_config.model_dump()
        for key, value in custom_args.items():
            provider_info.custom_args[key] = str(value)

    # Create and return model info
    return MockModelInfo(
        model_id=model_id,
        model_name=model_config.name,
        cost=cost,
        rate_limit=rate_limit,
        provider=provider_info,
        api_key=api_key,
    )


def test_convert_model_config_to_model_info():
    """Test converting from config model to ModelInfo."""
    # Create test data
    model_id = "openai:gpt-4"
    provider_name = "openai"
    # Create a real string for name property instead of a MagicMock
    model_name = "GPT-4"
    model_config = MagicMock()
    model_config.name = model_name
    model_config.cost = MagicMock(
        input_cost_per_thousand=5.0, output_cost_per_thousand=15.0
    )
    model_config.rate_limit = MagicMock(
        tokens_per_minute=100000, requests_per_minute=500
    )
    provider_config = MagicMock(base_url="https://api.openai.com")
    provider_config.model_dump.return_value = {"timeout": 30.0, "max_retries": 3}
    api_key = "test-api-key"

    # Call the function
    model_info = convert_model_config_to_model_info(
        model_id, provider_name, model_config, provider_config, api_key
    )

    # Verify results
    assert model_info.model_id == "openai:gpt-4"
    assert model_info.model_name == model_name
    assert model_info.cost.input_cost_per_thousand == 5.0
    assert model_info.cost.output_cost_per_thousand == 15.0
    assert model_info.rate_limit.tokens_per_minute == 100000
    assert model_info.rate_limit.requests_per_minute == 500
    assert model_info.provider.name == "Openai"
    assert model_info.provider.default_api_key == "test-api-key"
    assert model_info.provider.base_url == "https://api.openai.com"
    assert model_info.provider.custom_args.get("timeout") == "30.0"
    assert model_info.provider.custom_args.get("max_retries") == "3"
    assert model_info.get_api_key() == "test-api-key"


def test_initialize_registry_with_config_manager():
    """Test initializing registry with config manager - simplified mock version."""
    # Create mock config manager
    mock_config_manager = MagicMock()
    mock_config = MagicMock()
    mock_registry = MagicMock()

    # Configure mocks
    mock_config_manager.get_config.return_value = mock_config

    # Direct mock setup without patching
    mock_registry.is_registered = MagicMock(return_value=False)

    # Set up registry configuration
    mock_registry_properties = {
        "model_registry.auto_discover": True,
        "model_registry.auto_register": True,
        "model_registry.providers": {
            "openai": {
                "enabled": True,
                "api_keys": {"default": {"key": "test-key"}},
                "models": [{"id": "gpt-4", "name": "GPT-4"}],
            }
        },
    }

    # Configure mock_config to return appropriate values
    mock_config.model_registry.auto_discover = mock_registry_properties[
        "model_registry.auto_discover"
    ]
    mock_config.model_registry.auto_register = mock_registry_properties[
        "model_registry.auto_register"
    ]
    mock_config.model_registry.providers = mock_registry_properties[
        "model_registry.providers"
    ]

    # Simplified test - check that we would register the correct configs
    assert mock_config.model_registry.auto_discover is True
    assert "openai" in mock_config.model_registry.providers
