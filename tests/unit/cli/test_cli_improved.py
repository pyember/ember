"""CLI tests using direct invocation."""

import json

import pytest
import yaml

from ember.cli.main import main
from tests.unit.cli.fake_cli_registry import FakeCLIModelRegistry


class TestCLIBasics:
    """Basic CLI functionality."""

    def test_no_command_shows_help(self, mock_cli_args, capsys):
        """No arguments shows help and returns 2."""
        mock_cli_args()

        ret = main()

        assert ret == 2
        captured = capsys.readouterr()
        assert "usage: ember" in captured.out
        assert "Commands" in captured.out

    def test_invalid_command(self, mock_cli_args, capsys):
        """Invalid command shows help."""
        mock_cli_args("invalid-command")

        # argparse raises SystemExit for invalid commands
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "invalid choice" in captured.err

    def test_interrupt_handling(self, mock_cli_args, monkeypatch):
        """SIGINT returns 130."""
        mock_cli_args("test")

        # Mock the test command to raise KeyboardInterrupt
        def mock_test(*args):
            raise KeyboardInterrupt()

        monkeypatch.setattr("ember.cli.main.cmd_test", mock_test)

        ret = main()
        assert ret == 130


class TestConfigureCommand:
    """Test configure command with tmp_ctx."""

    def test_configure_get(self, tmp_ctx, mock_cli_args, capsys, monkeypatch):
        """Get configuration value."""
        # Set a value
        tmp_ctx.set_config("test.key", "test-value")
        tmp_ctx.save()

        # Make CLI use our test context
        monkeypatch.setattr("ember._internal.context.EmberContext.current", lambda: tmp_ctx)

        # Get via CLI
        mock_cli_args("configure", "get", "test.key")
        ret = main()

        assert ret == 0
        assert "test-value" in capsys.readouterr().out

    def test_configure_set(self, tmp_ctx, mock_cli_args, capsys, monkeypatch):
        """Set configuration value."""
        # Make CLI use our test context
        monkeypatch.setattr("ember._internal.context.EmberContext.current", lambda: tmp_ctx)

        mock_cli_args("configure", "set", "new.key", "new-value")
        ret = main()

        assert ret == 0
        output = capsys.readouterr().out
        assert "Updated new.key" in output

        # Verify it was saved
        assert tmp_ctx.get_config("new.key") == "new-value"

    @pytest.mark.parametrize("format_type", ["yaml", "json"])
    def test_configure_list_formats(self, tmp_ctx, mock_cli_args, capsys, format_type, monkeypatch):
        """List configuration in different formats."""
        # Set some config
        tmp_ctx.set_config("models.default", "gpt-4")
        tmp_ctx.set_config("features.streaming", True)
        tmp_ctx.save()

        # Make CLI use our test context
        monkeypatch.setattr("ember._internal.context.EmberContext.current", lambda: tmp_ctx)

        # List via CLI
        mock_cli_args("configure", "list", "--format", format_type)
        ret = main()

        assert ret == 0
        output = capsys.readouterr().out

        # Parse output
        if format_type == "json":
            data = json.loads(output)
        else:
            data = yaml.safe_load(output)

        assert data["models"]["default"] == "gpt-4"
        assert data["features"]["streaming"] is True


class TestSetupCommand:
    """Test setup wizard command."""

    def test_no_npm_error(self, tmp_ctx, mock_cli_args, capsys, monkeypatch):
        """Missing npm shows helpful error."""
        # Mock shutil.which to return None
        monkeypatch.setattr("shutil.which", lambda x: None)

        mock_cli_args("setup")
        ret = main()

        assert ret == 1
        output = capsys.readouterr().out
        assert "npm/npx is required" in output
        assert "https://nodejs.org/" in output


class TestModelsCommand:
    """Test models listing command."""

    @pytest.mark.skip(
        reason="CLI initializes provider clients before monkeypatch takes effect, "
        "requires API credentials. See test_cli_commands.py for proper mock approach."
    )
    def test_list_providers(self, tmp_ctx, mock_cli_args, capsys, monkeypatch):
        """List available providers."""
        # Mock the models API
        mock_providers = ["openai", "anthropic", "google"]

        class MockModels:
            def providers(self):
                return mock_providers

        monkeypatch.setattr("ember.api.models", MockModels())

        mock_cli_args("models", "--providers")
        ret = main()

        assert ret == 0
        output = capsys.readouterr().out
        for provider in mock_providers:
            assert provider in output


class TestConnectionTest:
    """Test the test command."""

    def test_successful_connection(self, tmp_ctx, mock_cli_args, capsys, monkeypatch):
        """Successful API connection."""
        # Set default model
        tmp_ctx.set_config("models.default", "test-model")
        tmp_ctx.save()

        # Make CLI use our test context
        monkeypatch.setattr("ember._internal.context.EmberContext.current", lambda: tmp_ctx)

        # Create test registry and set it on the context
        test_registry = FakeCLIModelRegistry()
        test_registry.set_model_response("test-model", "Hello from test model!")

        # Monkey patch the registry methods we need
        monkeypatch.setattr(tmp_ctx.model_registry, "get_model", test_registry.get_model)
        monkeypatch.setattr(tmp_ctx.model_registry, "invoke_model", test_registry.invoke_model)

        mock_cli_args("test")
        ret = main()

        assert ret == 0
        captured = capsys.readouterr()
        assert "Testing connection with test-model" in captured.out
        assert "Success:" in captured.out
        assert "Hello from test model!" in captured.out

    def test_failed_connection(self, tmp_ctx, mock_cli_args, capsys, monkeypatch):
        """Failed API connection shows error."""
        # Make CLI use our test context
        monkeypatch.setattr("ember._internal.context.EmberContext.current", lambda: tmp_ctx)

        # Create test registry that fails
        test_registry = FakeCLIModelRegistry()
        test_registry.set_model_error("gpt-3.5-turbo", Exception("No API key"))

        # Monkey patch the registry methods
        monkeypatch.setattr(tmp_ctx.model_registry, "get_model", test_registry.get_model)
        monkeypatch.setattr(tmp_ctx.model_registry, "invoke_model", test_registry.invoke_model)

        mock_cli_args("test")
        ret = main()

        assert ret == 1
        captured = capsys.readouterr()
        assert "Error: No API key" in captured.err


class TestVersionCommand:
    """Test version display."""

    def test_version_display(self, tmp_ctx, mock_cli_args, capsys, monkeypatch):
        """Version command shows version."""
        # Mock the version
        monkeypatch.setattr("ember.__version__", "1.2.3")

        mock_cli_args("version")
        ret = main()

        assert ret == 0
        assert "Ember 1.2.3" in capsys.readouterr().out


@pytest.mark.parametrize(
    "exit_code,error_msg",
    [
        (1, "General error"),
        (2, "Invalid usage"),
        (130, "Interrupted"),
    ],
)
def test_error_codes(tmp_ctx, mock_cli_args, monkeypatch, exit_code, error_msg):
    """Various error conditions return correct codes."""
    mock_cli_args("test")

    def mock_cmd(args):
        if exit_code == 130:
            raise KeyboardInterrupt()
        elif exit_code == 2:
            raise SystemExit(2)
        else:
            raise Exception(error_msg)

    monkeypatch.setattr("ember.cli.main.cmd_test", mock_cmd)

    ret = main()
    assert ret == exit_code
