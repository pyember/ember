"""Integration tests for CLI and Context system.

Tests the full stack integration between:
- CLI commands
- Context system
- Configuration persistence
- Credential management
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from ember._internal.context import EmberContext
from ember.cli.main import main


@pytest.fixture(autouse=True)
def clean_context():
    """Clean context singleton before each test."""
    # Clear any existing singleton
    if hasattr(EmberContext._thread_local, "context"):
        delattr(EmberContext._thread_local, "context")
    EmberContext._context_var.set(None)
    yield
    # Clean up after test
    if hasattr(EmberContext._thread_local, "context"):
        delattr(EmberContext._thread_local, "context")
    EmberContext._context_var.set(None)


@pytest.fixture
def isolated_env():
    """Create isolated environment for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override config location
        with patch.dict(
            os.environ,
            {
                "EMBER_CONFIG_PATH": str(Path(tmpdir) / ".ember"),
                "HOME": tmpdir,  # For fallback
            },
            clear=True,
        ):
            yield Path(tmpdir)


class TestCLIContextIntegration:
    """Full CLI to Context integration."""

    def test_configure_persistence(self, isolated_env):
        """Configuration persists across invocations."""
        # Set config via CLI
        with patch("sys.argv", ["ember", "configure", "set", "models.default", "gpt-4"]):
            assert main() == 0

        # Verify file created
        config_file = isolated_env / ".ember" / "config.yaml"
        assert config_file.exists()

        # Read via new context
        ctx = EmberContext(isolated=True)
        assert ctx.get_config("models.default") == "gpt-4"

        # Get via CLI should match
        with patch("sys.argv", ["ember", "configure", "get", "models.default"]):
            with patch("sys.stdout.write") as mock_write:
                assert main() == 0
                # Check output contains gpt-4
                output = "".join(call.args[0] for call in mock_write.call_args_list)
                assert "gpt-4" in output

    def test_credential_flow(self, isolated_env):
        """Credential save and use flow."""
        # Save credential via API
        ctx = EmberContext.current()
        ctx._credentials.save_api_key("openai", "test-key-123")

        # Should be available in new context
        new_ctx = EmberContext(isolated=True)
        assert new_ctx.get_credential("openai", "OPENAI_API_KEY") == "test-key-123"

        # Test command should use it
        with patch("sys.argv", ["ember", "test", "--model", "gpt-3.5-turbo"]):
            with patch.object(new_ctx.model_registry, "invoke_model") as mock_invoke:
                mock_invoke.return_value.data = "Success"

                # Context should have access to credential
                key = new_ctx.get_credential("openai", "OPENAI_API_KEY")
                assert key == "test-key-123"

    @unittest.skipIf(sys.platform.startswith("win"), "disabled on Windows")
    def test_migration_integration(self, isolated_env):
        """Migration updates context correctly."""
        # Create old-style config
        old_dir = isolated_env / ".ember"
        old_dir.mkdir()

        old_config = {
            "version": "1.0",
            "providers": {"openai": {"default_model": "gpt-4"}},
            "models": {"default": "gpt-3.5-turbo"},
        }

        with open(old_dir / "config.json", "w") as f:
            json.dump(old_config, f)

        # Run migration
        with patch("sys.argv", ["ember", "configure", "migrate"]):
            assert main() == 0

        # Verify migrated correctly
        ctx = EmberContext.current()
        ctx.reload()
        assert ctx.get_config("models.default") == "gpt-3.5-turbo"
        assert ctx.get_config("providers.openai.default_model") == "gpt-4"

        # Old file should be backed up
        assert not (old_dir / "config.json").exists()
        assert list(old_dir.glob("config.json.bak.*"))


class TestSetupWizardIntegration:
    """Setup wizard integration with context."""

    def test_wizard_updates_context(self):
        """Setup wizard changes are reflected in context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / ".ember"

            # Mock npx to simulate wizard saving config
            def mock_npx(*args, **kwargs):
                # Wizard writes config + credentials into ~/.ember/config.yaml
                config_path.mkdir(exist_ok=True)
                config_file = config_path / "config.yaml"
                config_file.write_text(
                    """
models:
  default: gpt-4
  temperature: 0.7
"""
                )

                from ember.core.credentials import CredentialManager

                mgr = CredentialManager(config_path)
                mgr.save_api_key("openai", "wizard-key")
                return subprocess.CompletedProcess(args, 0)

            with patch.dict(os.environ, {"EMBER_CONFIG_PATH": str(config_path)}):
                with patch("subprocess.run", side_effect=mock_npx):
                    with patch("shutil.which", return_value="/usr/bin/npx"):
                        # Run setup
                        with patch("sys.argv", ["ember", "setup"]):
                            assert main() == 0

                        # Context should have wizard's changes
                        ctx = EmberContext.current()
                        assert ctx.get_config("models.default") == "gpt-4"
                        assert ctx.get_credential("openai", "OPENAI_API_KEY") == "wizard-key"


class TestEndToEndWorkflows:
    """Complete user workflows."""

    def test_first_time_user_flow(self, isolated_env):
        """New user setup and first API call."""
        # 1. User runs setup
        with patch("shutil.which", return_value=None):  # No npm
            with patch("sys.argv", ["ember", "setup"]):
                ret = main()
                assert ret == 1  # Expected failure

        # 2. User sets config manually
        with patch("sys.argv", ["ember", "configure", "set", "models.default", "gpt-3.5-turbo"]):
            assert main() == 0

        # 3. User saves API key
        ctx = EmberContext.current()
        ctx._credentials.save_api_key("openai", "user-api-key")

        # 4. User tests connection
        with patch("sys.argv", ["ember", "test"]):
            with patch.object(ctx.model_registry, "invoke_model") as mock:
                mock.return_value.data = "Hello!"
                assert main() == 0

                # Should use configured model
                mock.assert_called_with("gpt-3.5-turbo", "Say hello!")

    def test_model_discovery_flow(self, isolated_env):
        """User discovers and uses models."""
        # 1. List providers
        with patch("sys.argv", ["ember", "models", "--providers"]):
            # models.providers is a bound method, need to patch differently
            with patch(
                "ember.api.models._global_models_api.providers",
                return_value=["openai", "anthropic"],
            ):
                assert main() == 0

        # 2. List OpenAI models
        with patch("sys.argv", ["ember", "models", "--provider", "openai"]):
            with patch("ember.api.models._global_models_api.discover") as mock_discover:
                mock_discover.return_value = {
                    "gpt-4": {"description": "Most capable"},
                    "gpt-3.5-turbo": {"description": "Fast and cheap"},
                }
                assert main() == 0

        # 3. Configure chosen model
        with patch("sys.argv", ["ember", "configure", "set", "models.default", "gpt-4"]):
            assert main() == 0

        # 4. Verify configuration
        ctx = EmberContext.current()
        assert ctx.get_config("models.default") == "gpt-4"


class TestErrorRecovery:
    """Error handling and recovery."""

    def test_corrupted_config_recovery(self, isolated_env):
        """System recovers from corrupted config."""
        config_path = isolated_env / ".ember"
        config_path.mkdir()

        # Write corrupted config
        config_file = config_path / "config.yaml"
        config_file.write_text("{ invalid: yaml: content }")

        # CLI should handle gracefully
        with patch("sys.argv", ["ember", "configure", "list"]):
            ret = main()
            assert ret == 1

        # Should be able to write new config
        with patch("sys.argv", ["ember", "configure", "set", "test", "value"]):
            assert main() == 0

        # Verify recovery
        ctx = EmberContext.current()
        ctx.reload()
        assert ctx.get_config("test") == "value"

    @unittest.skipIf(sys.platform.startswith("win"), "disabled on Windows")
    def test_missing_credentials_handling(self, isolated_env):
        """Graceful handling of missing credentials."""
        # No credentials set
        with patch("sys.argv", ["ember", "test"]):
            with patch.dict(os.environ, {}, clear=True):
                ctx = EmberContext.current()

                # Should fail gracefully
                with patch.object(ctx.model_registry, "invoke_model") as mock:
                    mock.side_effect = Exception("API key required")
                    ret = main()
                    assert ret == 1  # Error exit code
