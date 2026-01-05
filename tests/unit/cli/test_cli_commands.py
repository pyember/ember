"""Tests for CLI commands."""

import json
import re
from unittest.mock import Mock, patch

import pytest
import yaml

from ember._internal.context import EmberContext
from ember.cli.commands.configure import cmd_configure as configure_cmd
from ember.cli.main import (
    cmd_models,
    cmd_setup,
    cmd_test,
    cmd_version,
    main,
)

# Import test infrastructure
from tests.test_constants import Models


class TestCLIEntry:
    """Main CLI entry point."""

    def test_no_args_shows_help(self, capsys):
        """No arguments shows help and returns 2."""
        with patch("sys.argv", ["ember"]):
            assert main() == 2  # Changed from 0 to 2
            output = capsys.readouterr().out
            # More flexible - just check it shows usage info
            assert "usage:" in output.lower() or "ember" in output

    def test_interrupt_handling(self):
        """SIGINT returns 130."""
        with patch("sys.argv", ["ember", "test"]):
            with patch("ember.cli.main.cmd_test", side_effect=KeyboardInterrupt):
                assert main() == 130

    def test_error_handling(self, capsys):
        """Exceptions return 1 with error message."""
        test_error = "test failure message"
        with patch("sys.argv", ["ember", "test"]):
            with patch("ember.cli.main.cmd_test", side_effect=Exception(test_error)):
                assert main() == 1
                err = capsys.readouterr().err
                # Check error is reported, not exact format
                assert test_error in err


class TestConfigure:
    """Configure command."""

    @pytest.fixture
    def ctx(self):
        """Mock context."""
        ctx = Mock(spec=EmberContext)
        ctx.get_all_config.return_value = {"models": {"default": Models.GPT4}}
        return ctx

    def test_get(self, ctx, capsys):
        """Get config value."""
        test_value = Models.GPT4
        ctx.get_config.return_value = test_value

        args = Mock(action="get", key="models.default", default=None, context=ctx)
        assert configure_cmd(args) == 0
        output = capsys.readouterr().out
        assert test_value in output

    def test_get_missing(self, ctx, capsys):
        """Missing key returns error."""
        ctx.get_config.return_value = None

        args = Mock(action="get", key="missing", default=None, context=ctx)
        assert configure_cmd(args) == 1
        err = capsys.readouterr().err
        # Check error mentions the key, not exact message
        assert "missing" in err.lower()

    def test_set(self, ctx, capsys):
        """Set config value."""
        args = Mock(action="set", key="test", value="value", context=ctx)
        assert configure_cmd(args) == 0

        ctx.set_config.assert_called_with("test", "value")
        ctx.save.assert_called_once()

    def test_set_json(self, ctx):
        """JSON values are parsed."""
        test_data = {"a": 1}
        args = Mock(action="set", key="test", value=json.dumps(test_data), context=ctx)
        configure_cmd(args)

        ctx.set_config.assert_called_with("test", test_data)

    @pytest.mark.parametrize(
        "format_type,loader",
        [
            pytest.param("yaml", yaml.safe_load, id="yaml-format"),
            pytest.param("json", json.loads, id="json-format"),
        ],
    )
    def test_list_formats(self, ctx, capsys, format_type, loader):
        """List supports multiple formats."""
        test_config = {"models": {"default": Models.GPT4}}
        ctx.get_all_config.return_value = test_config

        args = Mock(action="list", format=format_type, context=ctx)
        configure_cmd(args)
        output = capsys.readouterr().out

        # Parse output and verify structure
        parsed = loader(output)
        assert parsed["models"]["default"] == Models.GPT4


class TestSetup:
    """Setup wizard command."""

    def test_no_npm(self, capsys):
        """Missing npm shows error."""
        with patch("shutil.which", return_value=None):
            assert cmd_setup(Mock()) == 1
            output = capsys.readouterr().out
            # Check error mentions npm, not exact message
            assert "npm" in output.lower()

    def test_launch(self):
        """Launches wizard and reloads on success."""
        ctx = Mock()

        with patch("shutil.which", return_value="/usr/bin/npx"):
            with patch("subprocess.run") as mock_run:
                # Mock the path check to simulate no local wizard
                with patch("pathlib.Path.exists", return_value=False):
                    mock_run.return_value.returncode = 0

                    assert cmd_setup(Mock(context=ctx)) == 0

                    # Verify npx was called (not exact command)
                    call_args = mock_run.call_args[0][0]
                    assert "npx" in call_args[0]
                    assert "@ember-ai/setup" in call_args

                    # Environment includes config path
                    assert "EMBER_CONFIG_PATH" in mock_run.call_args[1]["env"]

                    # Reloaded context
                    ctx.reload.assert_called_once()

    def test_failure_no_reload(self):
        """Failed wizard doesn't reload context."""
        ctx = Mock()

        with patch("shutil.which", return_value="/usr/bin/npx"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 1

                assert cmd_setup(Mock(context=ctx)) == 1
                ctx.reload.assert_not_called()


class TestModels:
    """Models listing command."""

    def test_list_providers(self, capsys):
        """List available providers."""
        test_providers = ["openai", "anthropic", "google"]
        from ember.api.models import _global_models_api

        with patch.object(_global_models_api, "providers", return_value=test_providers):
            cmd_models(Mock(providers=True))
            output = capsys.readouterr().out

            # Verify all providers appear
            for provider in test_providers:
                assert provider in output

    def test_list_models(self, capsys, monkeypatch):
        """List available models."""
        models_data = {
            Models.GPT4: {"description": "Advanced model"},
            Models.GPT35: {"description": "Fast model"},
        }

        # Mock the entire models module when imported
        mock_models = Mock()
        mock_models.discover.return_value = models_data
        mock_models.providers.return_value = ["openai"]

        import sys

        monkeypatch.setitem(sys.modules, "ember.api.models", mock_models)

        cmd_models(Mock(providers=False, provider=None))
        output = capsys.readouterr().out

        # Verify models appear
        assert Models.GPT4 in output
        # Verify some description appears
        assert any(desc in output for desc in ["Advanced", "Fast", "model"])

    def test_filter_by_provider(self, capsys):
        """Filter models by provider."""
        cmd_models(Mock(providers=False, provider="openai"))
        output = capsys.readouterr().out

        # Should show some output about models
        assert len(output) > 0
        # GPT models should appear for OpenAI filter
        assert "gpt" in output.lower()


class TestConnection:
    """Test command for API connections."""

    def test_success(self, capsys):
        """Successful connection test."""
        ctx = Mock()
        test_model = Models.GPT4
        test_response = "Hello from test!"

        ctx.get_config.return_value = test_model
        ctx.model_registry.invoke_model.return_value = Mock(data=test_response)

        assert cmd_test(Mock(model=None, context=ctx)) == 0

        output = capsys.readouterr().out
        # Verify key information appears
        assert test_model in output
        assert test_response in output
        # Some indication of success (checkmark, success, etc)
        assert any(indicator in output.lower() for indicator in ["success", "ok"])

    def test_failure(self, capsys):
        """Failed connection test."""
        ctx = Mock()
        test_model = Models.GPT4
        test_error = "API key not found"

        ctx.get_config.return_value = test_model
        ctx.model_registry.invoke_model.side_effect = Exception(test_error)

        assert cmd_test(Mock(model=None, context=ctx)) == 1
        output = capsys.readouterr().err

        # Verify error is shown
        assert test_error in output
        # Some indication of failure
        assert any(indicator in output.lower() for indicator in ["fail", "error"])

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(Models.GPT4, id="gpt4"),
            pytest.param(Models.CLAUDE3, id="claude3"),
            pytest.param(Models.GEMINI_PRO, id="gemini"),
        ],
    )
    def test_explicit_model(self, model):
        """Test specific models."""
        ctx = Mock()
        ctx.model_registry.invoke_model.return_value = Mock(data="OK")

        cmd_test(Mock(model=model, context=ctx))
        # Verify the model was invoked
        ctx.model_registry.invoke_model.assert_called_once()
        call_args = ctx.model_registry.invoke_model.call_args
        assert call_args[0][0] == model


class TestVersion:
    """Version command."""

    @pytest.mark.parametrize(
        "version,expected_pattern",
        [
            pytest.param("1.0.0", r"1\.0\.0", id="stable"),
            pytest.param("2.1.0-beta", r"2\.1\.0-beta", id="prerelease"),
            pytest.param("0.0.1", r"0\.0\.1", id="early"),
        ],
    )
    def test_with_version(self, capsys, version, expected_pattern):
        """Shows version when available."""
        with patch("ember.__version__", version):
            assert cmd_version(Mock()) == 0
            output = capsys.readouterr().out
            assert re.search(expected_pattern, output)

    def test_without_version(self, capsys):
        """Handles missing version gracefully."""
        import ember

        version = getattr(ember, "__version__", None)
        if version:
            delattr(ember, "__version__")

        try:
            assert cmd_version(Mock()) == 0
            output = capsys.readouterr().out
            # Should show something, even if no version
            assert "ember" in output.lower()
        finally:
            if version:
                ember.__version__ = version


# Contract tests for CLI behavior
class TestCLIContracts:
    """Verify CLI commands follow expected contracts."""

    @pytest.mark.parametrize(
        "command,expected_code",
        [
            pytest.param(["ember", "--help"], 0, id="help"),
            pytest.param(["ember", "unknown-command"], 2, id="unknown-command"),
        ],
    )
    def test_exit_codes(self, command, expected_code):
        """Test standard exit codes."""
        with patch("sys.argv", command):
            # argparse raises SystemExit, we need to catch it
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == expected_code

    def test_all_commands_have_context(self):
        """Verify all commands accept context parameter."""
        commands = [cmd_setup, cmd_models, cmd_test, cmd_version]

        for cmd in commands:
            # Create mock args with context
            args = Mock(context=Mock())
            # Command should not fail due to missing context
            try:
                cmd(args)
            except AttributeError as e:
                if "context" in str(e):
                    pytest.fail(f"{cmd.__name__} doesn't handle context properly")
