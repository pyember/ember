"""Tests for configuration import command."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from ember.cli.commands.config_import import (
    _backup_config,
    _find_external_config,
    _migrate_config,
    cmd_import,
)


class TestFindExternalConfig:
    """Test external config file discovery."""

    def test_find_yaml_config(self, tmp_path, monkeypatch):
        """Test finding YAML external config."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        external_dir = tmp_path / ".config/openai"
        external_dir.mkdir(parents=True)

        config_file = external_dir / "config.yaml"
        config_file.write_text("model: o4-mini")

        found = _find_external_config()
        assert found == config_file

    def test_find_json_config(self, tmp_path, monkeypatch):
        """Test finding JSON external config."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        external_dir = tmp_path / ".config/openai"
        external_dir.mkdir(parents=True)

        config_file = external_dir / "config.json"
        config_file.write_text('{"model": "o4-mini"}')

        found = _find_external_config()
        assert found == config_file

    def test_yaml_preferred_over_json(self, tmp_path, monkeypatch):
        """Test YAML is preferred when both exist."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        external_dir = tmp_path / ".config/openai"
        external_dir.mkdir(parents=True)

        yaml_file = external_dir / "config.yaml"
        yaml_file.write_text("model: yaml")

        json_file = external_dir / "config.json"
        json_file.write_text('{"model": "json"}')

        found = _find_external_config()
        assert found == yaml_file

    def test_no_external_dir(self, tmp_path, monkeypatch):
        """Test when .config/openai directory doesn't exist."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        found = _find_external_config()
        assert found is None

    def test_no_config_files(self, tmp_path, monkeypatch):
        """Test when .config/openai exists but has no config files."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        external_dir = tmp_path / ".config/openai"
        external_dir.mkdir(parents=True)

        found = _find_external_config()
        assert found is None


class TestMigrateConfig:
    """Test configuration migration."""

    def test_migrate_basic_config(self):
        """Test migrating basic external config."""
        external_config = {
            "model": "o4-mini",
            "provider": "openai",
            "providers": {
                "openai": {
                    "name": "OpenAI",
                    "baseURL": "https://api.openai.com/v1",
                    "envKey": "OPENAI_API_KEY",
                }
            },
        }

        migrated = _migrate_config(external_config)

        assert migrated["version"] == "1.0"
        assert migrated["model"] == "o4-mini"
        assert migrated["provider"] == "openai"
        assert migrated["providers"]["openai"]["env_key"] == "OPENAI_API_KEY"
        assert migrated["providers"]["openai"]["base_url"] == "https://api.openai.com/v1"
        assert "_migration" in migrated

    def test_migrate_with_external_fields(self):
        """Test migration preserves external-specific fields."""
        external_config = {
            "model": "o4-mini",
            "approvalMode": "suggest",
            "fullAutoErrorMode": "ask-user",
            "notify": True,
            "providers": {},
        }

        migrated = _migrate_config(external_config)

        assert migrated["_migration"]["original_fields"]["approvalMode"] == "suggest"
        assert migrated["_migration"]["original_fields"]["fullAutoErrorMode"] == "ask-user"
        assert migrated["_migration"]["original_fields"]["notify"] is True

    def test_migrate_multiple_providers(self):
        """Test migrating multiple providers."""
        external_config = {
            "providers": {
                "openai": {
                    "name": "OpenAI",
                    "baseURL": "https://api.openai.com/v1",
                    "envKey": "OPENAI_API_KEY",
                },
                "anthropic": {
                    "name": "Anthropic",
                    "baseURL": "https://api.anthropic.com/v1",
                    "envKey": "ANTHROPIC_API_KEY",
                },
            }
        }

        migrated = _migrate_config(external_config)

        assert len(migrated["providers"]) == 2
        assert migrated["providers"]["openai"]["env_key"] == "OPENAI_API_KEY"
        assert migrated["providers"]["anthropic"]["env_key"] == "ANTHROPIC_API_KEY"


class TestBackupConfig:
    """Test configuration backup."""

    def test_backup_creates_file(self, tmp_path):
        """Test backup creates a copy with timestamp."""
        original = tmp_path / "config.yaml"
        original.write_text("test: config")

        backup = _backup_config(original)

        assert backup.exists()
        assert backup != original
        assert ".backup.yaml" in str(backup)
        assert backup.read_text() == "test: config"

    def test_backup_preserves_format(self, tmp_path):
        """Test backup preserves file format."""
        original = tmp_path / "config.json"
        original.write_text('{"test": "config"}')

        backup = _backup_config(original)

        assert backup.suffix == ".json"
        assert ".backup.json" in str(backup)


class TestImportCommand:
    """Test the main import command."""

    def test_import_success(self, tmp_path, monkeypatch):
        """Test successful import."""
        # Setup mock paths
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Create external config
        external_dir = tmp_path / ".config/openai"
        external_dir.mkdir(parents=True)

        external_config = {
            "model": "o4-mini",
            "provider": "openai",
            "providers": {
                "openai": {
                    "name": "OpenAI",
                    "baseURL": "https://api.openai.com/v1",
                    "envKey": "OPENAI_API_KEY",
                }
            },
        }

        external_file = external_dir / "config.json"
        external_file.write_text(json.dumps(external_config))

        # Create mock args
        args = MagicMock()
        args.config_path = None
        args.output_path = None
        args.backup = True
        args.dry_run = False

        # Mock EmberContext.get_config_path
        ember_config = tmp_path / ".ember" / "config.yaml"
        with patch(
            "ember.cli.commands.config_import.EmberContext.get_config_path",
            return_value=ember_config,
        ):
            cmd_import(args)

        # Verify output file created
        assert ember_config.exists()

        # Verify content
        imported = yaml.safe_load(ember_config.read_text())
        assert imported["model"] == "o4-mini"
        assert imported["providers"]["openai"]["env_key"] == "OPENAI_API_KEY"

    def test_import_dry_run(self, tmp_path, monkeypatch, capsys):
        """Test dry run mode."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Create external config
        external_dir = tmp_path / ".config/openai"
        external_dir.mkdir(parents=True)

        external_config = {"model": "o4-mini"}
        external_file = external_dir / "config.yaml"
        external_file.write_text(yaml.dump(external_config))

        # Create mock args
        args = MagicMock()
        args.config_path = None
        args.output_path = None
        args.backup = True
        args.dry_run = True

        ember_config = tmp_path / ".ember" / "config.yaml"
        with patch(
            "ember.cli.commands.config_import.EmberContext.get_config_path",
            return_value=ember_config,
        ):
            cmd_import(args)

        # Verify no file created
        assert not ember_config.exists()

        # Verify output shown
        captured = capsys.readouterr()
        assert "Migrated configuration (dry run):" in captured.out
        assert "model: o4-mini" in captured.out

    def test_import_with_backup(self, tmp_path, monkeypatch):
        """Test import with existing config backup."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Create existing Ember config
        ember_dir = tmp_path / ".ember"
        ember_dir.mkdir(parents=True)

        existing_config = {"existing": "config"}
        ember_config = ember_dir / "config.yaml"
        ember_config.write_text(yaml.dump(existing_config))

        # Create external config
        external_dir = tmp_path / ".config/openai"
        external_dir.mkdir(parents=True)

        external_config = {"model": "o4-mini"}
        external_file = external_dir / "config.yaml"
        external_file.write_text(yaml.dump(external_config))

        # Create mock args
        args = MagicMock()
        args.config_path = None
        args.output_path = None
        args.backup = True
        args.dry_run = False

        with patch(
            "ember.cli.commands.config_import.EmberContext.get_config_path",
            return_value=ember_config,
        ):
            cmd_import(args)

        # Verify backup created
        backups = list(ember_dir.glob("config.*.backup.yaml"))
        assert len(backups) == 1

        # Verify backup content
        backup_content = yaml.safe_load(backups[0].read_text())
        assert backup_content == existing_config
