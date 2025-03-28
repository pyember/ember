"""Integration tests for the centralized configuration system.

This module verifies that the configuration system properly integrates
with the model registry and other components.
"""

import os
import tempfile

import pytest
import yaml

from ember.core.app_context import create_ember_app
from ember.core.registry.model.initialization import initialize_registry


@pytest.fixture
def test_config():
    """Create a temporary test configuration file."""
    config = {
        "registry": {
            "auto_discover": False,  # Disable to prevent API calls
            "auto_register": True,
            "providers": {
                "openai": {
                    "enabled": True,
                    "api_keys": {"default": {"key": "test-openai-key"}},
                    "models": {
                        "gpt-4": {
                            "id": "gpt-4",
                            "name": "GPT-4",
                            "provider": "openai",
                            "cost_input": 5.0,
                            "cost_output": 15.0,
                            "tokens_per_minute": 100000,
                            "requests_per_minute": 500,
                        }
                    },
                },
                "anthropic": {
                    "enabled": True,
                    "api_keys": {"default": {"key": "test-anthropic-key"}},
                    "models": {
                        "claude-3": {
                            "id": "claude-3",
                            "name": "Claude 3",
                            "provider": "anthropic",
                            "cost_input": 15.0,
                            "cost_output": 75.0,
                            "tokens_per_minute": 50000,
                            "requests_per_minute": 100,
                        }
                    },
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


# Config schema has been updated
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


# Config schema has been updated
def test_app_context_integration(test_config, monkeypatch):
    """Test that the app context properly loads and uses the configuration."""
    # Set config path via environment
    monkeypatch.setenv("EMBER_CONFIG", test_config)

    # Create app context
    app = create_ember_app()

    # Verify config was loaded - handle registry namespace
    config = app.config_manager.get_config()
    assert config.registry.auto_discover is False
    assert config.registry.auto_register is True

    # Verify model registry was initialized
    model_ids = app.model_registry.list_models()
    assert "openai:gpt-4" in model_ids
    assert "anthropic:claude-3" in model_ids

    # Verify models were loaded correctly
    openai_model = app.model_registry.get_model_info("openai:gpt-4")
    assert openai_model.model_name == "GPT-4"
    assert openai_model.get_api_key() == "test-openai-key"


# Config schema has been updated
def test_environment_variable_substitution(monkeypatch):
    """Test that environment variables are properly substituted in the configuration."""
    # Set environment variables
    monkeypatch.setenv("TEST_OPENAI_KEY", "openai-key-from-env")
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "anthropic-key-from-env")

    # Create config with environment variable references
    config = {
        "registry": {
            "auto_discover": False,
            "auto_register": True,
            "providers": {
                "openai": {
                    "enabled": True,
                    "api_keys": {"default": {"key": "${TEST_OPENAI_KEY}"}},
                    "models": {
                        "gpt-4": {
                            "id": "gpt-4",
                            "name": "GPT-4",
                            "provider": "openai",
                            "cost_input": 5.0,
                            "cost_output": 15.0,
                        }
                    },
                },
                "anthropic": {
                    "enabled": True,
                    "api_keys": {"default": {"key": "${TEST_ANTHROPIC_KEY}"}},
                    "models": {
                        "claude-3": {
                            "id": "claude-3",
                            "name": "Claude 3",
                            "provider": "anthropic",
                            "cost_input": 15.0,
                            "cost_output": 75.0,
                        }
                    },
                },
            },
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        # Initialize registry from config with auto-register explicitly enabled
        registry = initialize_registry(
            config_path=config_path, auto_discover=False, force_discovery=False
        )

        # Print debug info
        print(f"Registry models: {registry.list_models()}")

        # Verify the models were registered
        model_ids = registry.list_models()
        assert (
            "openai:gpt-4" in model_ids
        ), f"OpenAI model not found in registry. Models: {model_ids}"

        # Verify environment variables were substituted in the model info
        openai_model = registry.get_model_info("openai:gpt-4")
        assert openai_model is not None, "OpenAI model info is None"

        # Verify API key substitution
        assert openai_model.get_api_key() == "openai-key-from-env"

        # Verify Anthropic model
        assert "anthropic:claude-3" in model_ids
        anthropic_model = registry.get_model_info("anthropic:claude-3")
        assert anthropic_model is not None
        assert anthropic_model.get_api_key() == "anthropic-key-from-env"
    finally:
        # Clean up
        os.unlink(config_path)
