"""Test the new context and registry CLI commands."""

import json
from unittest.mock import Mock, patch

import pytest
import yaml

from ember.cli.main import main
from ember.core.credentials import CredentialNotFoundError


class TestContextViewCommand:
    """Test ember context view command."""

    def test_context_view_yaml(self, tmp_ctx, mock_cli_args, capsys):
        """Test viewing context as YAML."""
        # Set some config
        tmp_ctx.set_config("models.default", "gpt-4")
        tmp_ctx.set_config("features.caching", True)

        mock_cli_args("context", "view")
        ret = main()

        assert ret == 0
        output = capsys.readouterr().out

        # Parse as YAML
        config = yaml.safe_load(output)
        assert config["models"]["default"] == "gpt-4"
        assert config["features"]["caching"] is True

    def test_context_view_json(self, tmp_ctx, mock_cli_args, capsys):
        """Test viewing context as JSON."""
        tmp_ctx.set_config("models.default", "claude-3")

        mock_cli_args("context", "view", "--format", "json")
        ret = main()

        assert ret == 0
        output = capsys.readouterr().out

        # Parse as JSON
        config = json.loads(output)
        assert config["models"]["default"] == "claude-3"

    def test_context_view_filtered(self, tmp_ctx, mock_cli_args, capsys):
        """Test viewing filtered context."""
        tmp_ctx.set_config("models.default", "gpt-4")
        tmp_ctx.set_config("models.temperature", 0.7)
        tmp_ctx.set_config("features.streaming", True)

        mock_cli_args("context", "view", "--filter", "models")
        ret = main()

        assert ret == 0
        output = capsys.readouterr().out

        # Should only show models section
        config = yaml.safe_load(output)
        assert "models" in config
        assert config["models"]["default"] == "gpt-4"
        assert config["models"]["temperature"] == 0.7
        assert "features" not in config

    def test_context_validate_success(self, tmp_ctx, mock_cli_args, capsys):
        """Test context validation with valid config."""
        # Mock credential and model availability
        tmp_ctx.get_credential = Mock(return_value="test-key")
        tmp_ctx.list_models = Mock(return_value=["gpt-4", "claude-3"])
        tmp_ctx.set_config("models.default", "gpt-4")

        mock_cli_args("context", "validate")
        ret = main()

        assert ret == 0
        assert "Configuration is valid" in capsys.readouterr().out

    def test_context_validate_missing_key(self, tmp_ctx, mock_cli_args, capsys):
        """Test context validation with missing API key."""
        # Mock missing credential
        tmp_ctx.get_credential = Mock(side_effect=CredentialNotFoundError("missing"))
        tmp_ctx.set_config("providers.openai", {"base_url": "test"})

        mock_cli_args("context", "validate")
        ret = main()

        assert ret == 1
        output = capsys.readouterr().out
        assert "Configuration issues found" in output
        assert "Missing API key for openai" in output


class TestRegistryCommands:
    """Test ember registry commands."""

    def test_registry_list_models(self, tmp_ctx, mock_cli_args, capsys):
        """Test listing models."""
        tmp_ctx.list_models = Mock(return_value=["gpt-4", "claude-3", "gemini-1.5-pro-latest"])

        mock_cli_args("registry", "list-models")
        ret = main()

        assert ret == 0
        output = capsys.readouterr().out
        assert "Available models:" in output
        assert "gpt-4" in output
        assert "claude-3" in output
        assert "gemini-1.5-pro-latest" in output

    def test_registry_list_models_filtered(self, tmp_ctx, mock_cli_args, capsys):
        """Test listing models filtered by provider."""
        tmp_ctx.list_models = Mock(return_value=["gpt-4", "gpt-3.5-turbo", "claude-3"])

        # Mock the catalog
        with patch(
            "ember.models.catalog.MODEL_CATALOG",
            {
                "gpt-4": {"provider": "openai"},
                "gpt-3.5-turbo": {"provider": "openai"},
                "claude-3": {"provider": "anthropic"},
            },
        ):
            mock_cli_args("registry", "list-models", "--provider", "openai")
            ret = main()

        assert ret == 0
        output = capsys.readouterr().out
        assert "gpt-4" in output
        assert "gpt-3.5-turbo" in output
        assert "claude-3" not in output

    def test_registry_list_models_verbose(self, tmp_ctx, mock_cli_args, capsys):
        """Test listing models with verbose output."""
        tmp_ctx.list_models = Mock(return_value=["gpt-4", "claude-3"])

        # Mock the catalog
        with patch(
            "ember.models.catalog.MODEL_CATALOG",
            {
                "gpt-4": {
                    "provider": "openai",
                    "description": "Most capable GPT-4 model",
                    "context_window": 8192,
                },
                "claude-3": {
                    "provider": "anthropic",
                    "description": "Claude 3 Opus",
                    "context_window": 200000,
                },
            },
        ):
            mock_cli_args("registry", "list-models", "--verbose")
            ret = main()

        assert ret == 0
        output = capsys.readouterr().out
        assert "OPENAI Models:" in output
        assert "ANTHROPIC Models:" in output
        assert "Most capable GPT-4 model" in output
        assert "Context: 8,192 tokens" in output
        assert "Context: 200,000 tokens" in output

    def test_registry_list_providers(self, tmp_ctx, mock_cli_args, capsys):
        """Test listing providers."""

        # Mock credentials - signature is get_credential(provider) only
        def mock_get_credential(provider):
            if provider == "openai":
                return "test-key"
            raise CredentialNotFoundError("missing")

        tmp_ctx.get_credential = mock_get_credential

        mock_cli_args("registry", "list-providers")
        ret = main()

        assert ret == 0
        output = capsys.readouterr().out
        assert "Provider Status:" in output
        assert "openai" in output
        assert "Configured" in output
        assert "anthropic" in output
        assert "Not configured" in output

    def test_registry_info(self, tmp_ctx, mock_cli_args, capsys):
        """Test model info command."""
        from ember.models.catalog import ModelInfo

        mock_info = ModelInfo(
            id="gpt-4",
            provider="openai",
            description="Most capable GPT-4 model",
            context_window=8192,
            capabilities=frozenset(["chat", "code", "reasoning"]),
        )
        mock_cost = {"input": 30.0, "output": 60.0, "context": 0.0}

        with (
            patch(
                "ember.cli.commands.registry.get_model_info", return_value=mock_info
            ),
            patch(
                "ember.models.pricing.get_model_cost", return_value=mock_cost
            ),
        ):
            mock_cli_args("registry", "info", "gpt-4")
            ret = main()

        assert ret == 0
        output = capsys.readouterr().out
        assert "Model: gpt-4" in output
        assert "Provider: openai" in output
        assert "Description: Most capable GPT-4 model" in output
        assert "Context Window: 8,192 tokens" in output
        assert "Cost: $30.0000 input / $60.0000 output per 1M tokens" in output
        assert "Capabilities: chat, code, reasoning" in output

    def test_registry_info_not_found(self, tmp_ctx, mock_cli_args, capsys):
        """Test model info for non-existent model."""
        with patch("ember.models.catalog.MODEL_CATALOG", {}):
            mock_cli_args("registry", "info", "nonexistent-model")
            ret = main()

        assert ret == 1
        assert "Model 'nonexistent-model' not found" in capsys.readouterr().out


@pytest.fixture
def tmp_ctx(tmp_path):
    """Create temporary context for testing."""
    from ember._internal.context import EmberContext

    config_file = tmp_path / "config.yaml"
    # Create empty config file
    config_file.write_text("# Empty config\n")
    ctx = EmberContext(config_path=config_file)
    return ctx


@pytest.fixture
def mock_cli_args(monkeypatch):
    """Mock command line arguments."""

    def _mock_args(*args):
        import sys

        monkeypatch.setattr(sys, "argv", ["ember"] + list(args))

    return _mock_args
