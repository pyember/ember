"""Integration tests for the centralized configuration system.

This module verifies that the configuration system properly integrates
with the model registry and other components.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from ember.core.app_context import create_ember_app, get_app_context
from ember.core.config.manager import create_config_manager
from ember.core.registry.model.initialization import initialize_registry


@pytest.fixture
def test_config():
    """Create a temporary test configuration file."""
    config = {
        "model_registry": {
            "auto_discover": False,  # Disable to prevent API calls
            "auto_register": True,
            "providers": {
                "openai": {
                    "enabled": True,
                    "api_keys": {"default": {"key": "test-openai-key"}},
                    "models": [
                        {
                            "id": "gpt-4",
                            "name": "GPT-4",
                            "provider": "openai",
                            "cost": {
                                "input_cost_per_thousand": 5.0,
                                "output_cost_per_thousand": 15.0,
                            },
                            "rate_limit": {
                                "tokens_per_minute": 100000,
                                "requests_per_minute": 500,
                            },
                        }
                    ],
                },
                "anthropic": {
                    "enabled": True,
                    "api_keys": {"default": {"key": "test-anthropic-key"}},
                    "models": [
                        {
                            "id": "claude-3",
                            "name": "Claude 3",
                            "provider": "anthropic",
                            "cost": {
                                "input_cost_per_thousand": 15.0,
                                "output_cost_per_thousand": 75.0,
                            },
                            "rate_limit": {
                                "tokens_per_minute": 50000,
                                "requests_per_minute": 100,
                            },
                        }
                    ],
                },
            },
        },
        "logging": {"level": "INFO"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    yield config_path

    # Clean up
    os.unlink(config_path)


@pytest.mark.skip(reason="Needs update for new config schema")
def test_config_model_registry_integration(test_config):
    """Test that the configuration system properly integrates with the model registry."""
    # Initialize registry from config
    registry = initialize_registry(config_path=test_config)

    # Verify models were loaded
    model_ids = registry.list_models()
    assert "openai:gpt-4" in model_ids
    assert "anthropic:claude-3" in model_ids

    # Verify model data
    openai_model = registry.get_model_info("openai:gpt-4")
    assert openai_model.model_name == "GPT-4"
    assert openai_model.cost.input_cost_per_thousand == 5.0
    assert openai_model.provider.name == "Openai"
    assert openai_model.get_api_key() == "test-openai-key"

    anthropic_model = registry.get_model_info("anthropic:claude-3")
    assert anthropic_model.model_name == "Claude 3"
    assert anthropic_model.cost.output_cost_per_thousand == 75.0
    assert anthropic_model.provider.name == "Anthropic"
    assert anthropic_model.get_api_key() == "test-anthropic-key"


@pytest.mark.skip(reason="Needs update for new config schema")
def test_app_context_integration(test_config, monkeypatch):
    """Test that the app context properly loads and uses the configuration."""
    # Set config path via environment
    monkeypatch.setenv("EMBER_CONFIG", test_config)

    # Create app context
    app = create_ember_app()

    # Verify config was loaded
    config = app.config_manager.get_config()
    assert config.model_registry.auto_discover is False
    assert config.model_registry.auto_register is True

    # Verify model registry was initialized
    model_ids = app.model_registry.list_models()
    assert "openai:gpt-4" in model_ids
    assert "anthropic:claude-3" in model_ids

    # Verify models were loaded correctly
    openai_model = app.model_registry.get_model_info("openai:gpt-4")
    assert openai_model.model_name == "GPT-4"
    assert openai_model.get_api_key() == "test-openai-key"


@pytest.mark.skip(reason="Needs update for new config schema")
def test_environment_variable_substitution(monkeypatch):
    """Test that environment variables are properly substituted in the configuration."""
    # Set environment variables
    monkeypatch.setenv("TEST_OPENAI_KEY", "openai-key-from-env")
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "anthropic-key-from-env")

    # Create config with environment variable references
    config = {
        "model_registry": {
            "auto_discover": False,
            "providers": {
                "openai": {
                    "enabled": True,
                    "api_keys": {"default": {"key": "${TEST_OPENAI_KEY}"}},
                    "models": [{"id": "gpt-4", "name": "GPT-4", "provider": "openai"}],
                },
                "anthropic": {
                    "enabled": True,
                    "api_keys": {"default": {"key": "${TEST_ANTHROPIC_KEY}"}},
                    "models": [
                        {"id": "claude-3", "name": "Claude 3", "provider": "anthropic"}
                    ],
                },
            },
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        # Initialize registry from config
        registry = initialize_registry(config_path=config_path)

        # Verify environment variables were substituted
        openai_model = registry.get_model_info("openai:gpt-4")
        assert openai_model.get_api_key() == "openai-key-from-env"

        anthropic_model = registry.get_model_info("anthropic:claude-3")
        assert anthropic_model.get_api_key() == "anthropic-key-from-env"
    finally:
        # Clean up
        os.unlink(config_path)
