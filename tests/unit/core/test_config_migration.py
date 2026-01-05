"""Test configuration migration."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from ember._internal.configuration.migrations import migrate_config, migrate_credentials
from ember.context import EmberContext


def test_migrate_credentials():
    """Test credentials migration from old format to config.yaml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create old credentials file
        old_dir = Path(tmpdir) / ".ember"
        old_dir.mkdir()
        old_creds = old_dir / "credentials"

        old_data = {"openai": {"api_key": "sk-test123", "created_at": "2024-01-01T00:00:00"}}

        with open(old_creds, "w") as f:
            json.dump(old_data, f)

        # Patch home directory and ensure no migration marker
        with patch.object(Path, "home", return_value=Path(tmpdir)):
            # Ensure no existing migration marker
            marker = old_dir / ".credentials_migrated"
            if marker.exists():
                marker.unlink()

            # Create isolated context for the migration
            ctx = EmberContext(isolated=True)
            ctx._config_file = old_dir / "config.yaml"

            with patch(
                "ember._internal.configuration.migrations.EmberContext.current",
                return_value=ctx,
            ):
                # Run migration
                result = migrate_credentials()

                assert result is True

                # Check old file was removed
                assert not old_creds.exists()

                # Check backup was created (with timestamp)
                backups = list(old_dir.glob("credentials.bak.*"))
                assert len(backups) == 1

                # Verify backup content
                with open(backups[0]) as f:
                    data = json.load(f)
                    assert "openai" in data
                    assert data["openai"]["api_key"] == "sk-test123"

                # Verify credential was migrated to config.yaml
                assert ctx.get_config("providers.openai.api_key") == "sk-test123"


def test_migrate_config():
    """Test config migration from JSON to YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create old config file
        old_dir = Path(tmpdir) / ".ember"
        old_dir.mkdir()
        old_config = old_dir / "config.json"

        old_data = {
            "version": "1.0",
            "providers": {"openai": {"default_model": "gpt-4"}},
            "models": {"default": "gpt-3.5-turbo"},
        }

        with open(old_config, "w") as f:
            json.dump(old_data, f)

        # Patch home directory
        with patch.object(Path, "home", return_value=Path(tmpdir)):
            # Create isolated context
            ctx = EmberContext(isolated=True)

            # Patch current context
            with patch(
                "ember._internal.configuration.migrations.EmberContext.current",
                return_value=ctx,
            ):
                # Run migration
                result = migrate_config()

                assert result is True

                # Check config was migrated (version field excluded)
                assert ctx.get_config("providers.openai.default_model") == "gpt-4"
                assert ctx.get_config("models.default") == "gpt-3.5-turbo"

                # Check old file was removed
                assert not old_config.exists()

                # Check backup was created (with timestamp)
                backups = list(old_dir.glob("config.json.bak.*"))
                assert len(backups) == 1


def test_no_migration_needed():
    """Test when no migration is needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # No old files exist
        with patch.object(Path, "home", return_value=Path(tmpdir)):
            assert migrate_credentials() is False
            assert migrate_config() is False
