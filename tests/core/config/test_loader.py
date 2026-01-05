"""Tests for enhanced configuration loader."""

import json

import pytest
import yaml

from ember.core.config.loader import load_config, save_config


class TestConfigLoader:
    """Test suite for ConfigLoader."""

    def test_load_yaml_config(self, tmp_path):
        """Test loading YAML configuration."""
        config = {
            "version": "1.0",
            "model": "gpt-4",
            "providers": {
                "openai": {
                    "api_key": "test-key",
                    "base_url": "https://api.openai.com/v1",
                }
            },
        }

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))

        loaded = load_config(config_file)
        assert loaded == config

    def test_load_json_config(self, tmp_path):
        """Test loading JSON configuration."""
        config = {
            "version": "1.0",
            "model": "gpt-4",
            "providers": {
                "openai": {
                    "api_key": "test-key",
                    "base_url": "https://api.openai.com/v1",
                }
            },
        }

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config, indent=2))

        loaded = load_config(config_file)
        assert loaded == config

    def test_load_extensionless_yaml(self, tmp_path):
        """Test loading YAML from extensionless file."""
        config = {"model": "gpt-4", "temperature": 0.7}

        config_file = tmp_path / "config"
        config_file.write_text(yaml.dump(config))

        loaded = load_config(config_file)
        assert loaded == config

    def test_load_extensionless_json(self, tmp_path):
        """Test loading JSON from extensionless file."""
        config = {"model": "gpt-4", "temperature": 0.7}

        config_file = tmp_path / "config"
        config_file.write_text(json.dumps(config))

        loaded = load_config(config_file)
        assert loaded == config

    def test_file_not_found(self):
        """Test handling of missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_invalid_format(self, tmp_path):
        """Test handling of invalid file content."""
        config_file = tmp_path / "config"
        config_file.write_text("invalid: [")

        with pytest.raises(ValueError):
            load_config(config_file)

    def test_empty_yaml(self, tmp_path):
        """Test loading empty YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        loaded = load_config(config_file)
        assert loaded == {}

    def test_save_yaml_config(self, tmp_path):
        """Test saving configuration as YAML."""
        config = {"version": "1.0", "providers": {"openai": {"api_key": "test"}}}

        config_file = tmp_path / "config.yaml"
        save_config(config, config_file)

        # Verify file was created
        assert config_file.exists()

        # Load and verify content
        loaded = yaml.safe_load(config_file.read_text())
        assert loaded == config

    def test_save_json_config(self, tmp_path):
        """Test saving configuration as JSON."""
        config = {"version": "1.0", "providers": {"openai": {"api_key": "test"}}}

        config_file = tmp_path / "config.json"
        save_config(config, config_file)

        # Verify file was created
        assert config_file.exists()

        # Load and verify content
        loaded = json.loads(config_file.read_text())
        assert loaded == config

    def test_save_creates_directory(self, tmp_path):
        """Test that save creates parent directories."""
        config = {"test": "value"}

        config_file = tmp_path / "nested" / "dir" / "config.yaml"
        save_config(config, config_file)

        assert config_file.exists()
        assert config_file.parent.exists()


class TestPlaceholderHandling:
    """Config loader preserves ${VAR} placeholders verbatim."""

    def test_load_config_preserves_placeholders_in_mapping(self, tmp_path, monkeypatch):
        monkeypatch.setenv("API_KEY", "secret-key-123")

        config = {"providers": {"openai": {"api_key": "${API_KEY}", "model": "gpt-4"}}}

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))

        loaded = load_config(config_file)
        assert loaded["providers"]["openai"]["api_key"] == "${API_KEY}"
        assert loaded["providers"]["openai"]["model"] == "gpt-4"

    def test_load_config_preserves_placeholders_in_list(self, tmp_path):
        config = {"items": ["${VAR1}", "static", "${VAR2}"]}

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))

        loaded = load_config(config_file)
        assert loaded["items"] == ["${VAR1}", "static", "${VAR2}"]


class TestRealWorldScenarios:
    """Test real-world configuration scenarios."""

    def test_codex_style_config(self, tmp_path, monkeypatch):
        """Test loading Codex-style configuration."""
        config = {
            "model": "o4-mini",
            "provider": "openai",
            "providers": {
                "openai": {
                    "name": "OpenAI",
                    "baseURL": "https://api.openai.com/v1",
                    "envKey": "OPENAI_API_KEY",
                    "api_key": "${OPENAI_API_KEY}",
                },
                "anthropic": {
                    "name": "Anthropic",
                    "baseURL": "https://api.anthropic.com/v1",
                    "envKey": "ANTHROPIC_API_KEY",
                    "api_key": "${ANTHROPIC_API_KEY}",
                },
            },
        }

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config, indent=2))

        loaded = load_config(config_file)
        assert loaded["providers"]["openai"]["api_key"] == "${OPENAI_API_KEY}"
        assert loaded["providers"]["anthropic"]["api_key"] == "${ANTHROPIC_API_KEY}"

    def test_mixed_format_config(self, tmp_path, monkeypatch):
        """Test configuration with mixed static and env var values."""
        config = {
            "environments": {
                "dev": {"api_key": "dev-static-key", "url": "http://localhost:8000"},
                "prod": {"api_key": "${PROD_KEY}", "url": "https://api.production.com"},
            }
        }

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))

        loaded = load_config(config_file)
        assert loaded["environments"]["dev"]["api_key"] == "dev-static-key"
        assert loaded["environments"]["prod"]["api_key"] == "${PROD_KEY}"
