"""Integration tests for configuration system."""

import json
from unittest.mock import MagicMock, patch

import pytest
import yaml

from ember._internal.context import EmberContext
from ember.core.config.compatibility_adapter import CompatibilityAdapter
from ember.core.config.loader import load_config
from ember.core.credentials import CredentialNotFoundError


class TestConfigurationIntegration:
    """Test full configuration system integration."""

    @pytest.fixture
    def external_config(self):
        """External tool configuration."""
        return {
            "model": "gpt-4",
            "provider": "openai",
            "approvalMode": "suggest",
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
            },
        }

    def test_external_config_direct_paste(self, tmp_path, monkeypatch):
        """Test user can paste external config and it works."""
        # User pastes external config
        config_file = tmp_path / ".ember" / "config.json"
        config_file.parent.mkdir(parents=True)

        external_config = {
            "model": "o4-mini",
            "provider": "openai",
            "approvalMode": "suggest",
            "providers": {
                "openai": {
                    "name": "OpenAI",
                    "baseURL": "https://api.openai.com/v1",
                    "envKey": "OPENAI_API_KEY",
                }
            },
        }

        config_file.write_text(json.dumps(external_config, indent=2))

        # Create context with this config
        monkeypatch.setattr(EmberContext, "get_config_path", staticmethod(lambda: config_file))

        ctx = EmberContext()

        # Verify it loaded and adapted correctly
        assert ctx.get_config("model") == "o4-mini"
        assert ctx.get_config("provider") == "openai"

        # Verify provider was adapted
        openai_config = ctx.get_config("providers.openai")
        assert openai_config["env_key"] == "OPENAI_API_KEY"
        assert openai_config["base_url"] == "https://api.openai.com/v1"

        # Verify credentials work
        with pytest.raises(CredentialNotFoundError):
            ctx.get_credential("openai", "OPENAI_API_KEY")

    def test_mixed_config_formats(self, tmp_path, monkeypatch):
        """Test loading configs in different formats."""
        configs = [
            ("yaml", "config.yaml", yaml.dump),
            ("json", "config.json", lambda x: json.dumps(x, indent=2)),
        ]

        for format_name, filename, dumper in configs:
            config_dir = tmp_path / format_name / ".ember"
            config_dir.mkdir(parents=True)
            config_file = config_dir / filename

            config = {
                "version": "1.0",
                "model": f"model-{format_name}",
                "providers": {
                    "test": {
                        "api_key": "${TEST_API_KEY}",
                        "base_url": "https://test.com",
                    }
                },
            }

            config_file.write_text(dumper(config))

            # Load and verify
            loaded = load_config(config_file)
            assert loaded["model"] == f"model-{format_name}"
            assert loaded["providers"]["test"]["api_key"] == "${TEST_API_KEY}"

    def test_environment_variable_precedence(self, tmp_path, monkeypatch):
        """Environment variables do not override centralized config credentials."""
        # Set env var
        monkeypatch.setenv("OPENAI_API_KEY", "env-key-123")

        # Create config with different key
        config_file = tmp_path / ".ember" / "config.yaml"
        config_file.parent.mkdir(parents=True)

        config = {
            "providers": {"openai": {"api_key": "config-key-456", "env_key": "OPENAI_API_KEY"}}
        }

        config_file.write_text(yaml.dump(config))

        # Create context
        monkeypatch.setattr(EmberContext, "get_config_path", staticmethod(lambda: config_file))
        ctx = EmberContext()

        # Config key should be used; env var is ignored (only used for warnings).
        cred = ctx.get_credential("openai", "OPENAI_API_KEY")
        assert cred == "config-key-456"

    @patch("ember.models.ModelRegistry")
    def test_model_loading_with_adapted_config(self, mock_registry, tmp_path, monkeypatch):
        """Test model loading works with adapted configuration."""
        # External config
        config_file = tmp_path / ".ember" / "config.json"
        config_file.parent.mkdir(parents=True)

        external_config = {
            "model": "gpt-4",
            "provider": "openai",
            "providers": {
                "openai": {
                    "name": "OpenAI",
                    "baseURL": "https://api.openai.com/v1",
                    "envKey": "OPENAI_API_KEY",
                }
            },
        }

        config_file.write_text(json.dumps(external_config))

        # Mock model registry
        mock_model = MagicMock()
        mock_registry.return_value.get_model.return_value = mock_model

        # Create context
        monkeypatch.setattr(EmberContext, "get_config_path", staticmethod(lambda: config_file))
        ctx = EmberContext()

        # Trigger model lookup
        ctx.get_model("gpt-4")

        # Verify registry was called correctly
        ctx.model_registry.get_model.assert_called_with("gpt-4")

    def test_backwards_compatibility(self, tmp_path, monkeypatch):
        """Test old Ember configs still work."""
        # Old-style Ember config
        config_file = tmp_path / ".ember" / "config.yaml"
        config_file.parent.mkdir(parents=True)

        old_config = {
            "version": "1.0",
            "models": {"default": "gpt-3.5-turbo"},
            "providers": {
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",
                    "base_url": "https://api.openai.com/v1",
                }
            },
        }

        config_file.write_text(yaml.dump(old_config))

        # Create context
        monkeypatch.setattr(EmberContext, "get_config_path", staticmethod(lambda: config_file))
        ctx = EmberContext()

        # Should work normally
        assert ctx.get_config("models.default") == "gpt-3.5-turbo"
        assert ctx.get_config("providers.openai.base_url") == "https://api.openai.com/v1"

    def test_config_migration_preserves_data(self, tmp_path, monkeypatch):
        """Test config migration preserves all data."""
        original = {
            "model": "o4-mini",
            "approvalMode": "full-auto",
            "custom_field": "preserved",
            "providers": {
                "openai": {
                    "name": "OpenAI",
                    "baseURL": "https://api.openai.com/v1",
                    "envKey": "OPENAI_API_KEY",
                    "custom_setting": "preserved",
                }
            },
        }

        # Adapt config
        adapted = CompatibilityAdapter.adapt_config(original)

        # Original fields preserved
        assert adapted["model"] == "o4-mini"
        assert adapted["custom_field"] == "preserved"

        # Provider adapted but custom fields preserved
        assert adapted["providers"]["openai"]["base_url"] == "https://api.openai.com/v1"
        assert adapted["providers"]["openai"]["custom_setting"] == "preserved"

        # External fields saved
        assert adapted["_external_compat"]["approvalMode"] == "full-auto"


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_multiple_provider_switching(self, tmp_path, monkeypatch):
        """Test switching between multiple providers."""
        config_file = tmp_path / ".ember" / "config.yaml"
        config_file.parent.mkdir(parents=True)

        config = {
            "provider": "openai",  # Default provider
            "providers": {
                "openai": {
                    "api_key": "sk-openai-123",
                    "base_url": "https://api.openai.com/v1",
                    "default_model": "gpt-4",
                },
                "anthropic": {
                    "api_key": "ant-claude-456",
                    "base_url": "https://api.anthropic.com/v1",
                    "default_model": "claude-3-opus",
                },
                "mistral": {
                    "api_key": "mst-789",
                    "base_url": "https://api.mistral.ai/v1",
                    "default_model": "mistral-large",
                },
            },
        }

        config_file.write_text(yaml.dump(config))
        monkeypatch.setattr(EmberContext, "get_config_path", staticmethod(lambda: config_file))

        # Test with different contexts
        ctx = EmberContext()

        # Default provider
        assert ctx.get_config("provider") == "openai"
        assert ctx.get_credential("openai", "OPENAI_API_KEY") == "sk-openai-123"

        # Switch provider via child context
        with ctx.create_child(provider="anthropic") as child:
            assert child.get_config("provider") == "anthropic"
            assert child.get_credential("anthropic", "ANTHROPIC_API_KEY") == "ant-claude-456"

        # Original context unchanged
        assert ctx.get_config("provider") == "openai"

    def test_dev_prod_environment_configs(self, tmp_path, monkeypatch):
        """Test different configs for dev/prod environments."""
        # Development config
        dev_config = {
            "environment": "development",
            "providers": {
                "openai": {
                    "api_key": "${OPENAI_DEV_KEY}",
                    "base_url": "https://api.openai.com/v1",
                    "timeout": 60,
                    "max_retries": 5,
                }
            },
            "logging": {"level": "DEBUG"},
        }

        # Production config
        prod_config = {
            "environment": "production",
            "providers": {
                "openai": {
                    "api_key": "${OPENAI_PROD_KEY}",
                    "base_url": "https://api.openai.com/v1",
                    "timeout": 30,
                    "max_retries": 3,
                }
            },
            "logging": {"level": "ERROR"},
        }

        # Test dev environment
        monkeypatch.setenv("EMBER_ENV", "development")

        dev_file = tmp_path / "dev" / ".ember" / "config.yaml"
        dev_file.parent.mkdir(parents=True)
        dev_file.write_text(yaml.dump(dev_config))

        monkeypatch.setattr(EmberContext, "get_config_path", staticmethod(lambda: dev_file))
        dev_ctx = EmberContext()

        assert dev_ctx.get_config("environment") == "development"
        assert dev_ctx.get_config("providers.openai.timeout") == 60
        assert dev_ctx.get_config("logging.level") == "DEBUG"

        # Test prod environment
        monkeypatch.setenv("EMBER_ENV", "production")

        prod_file = tmp_path / "prod" / ".ember" / "config.yaml"
        prod_file.parent.mkdir(parents=True)
        prod_file.write_text(yaml.dump(prod_config))

        monkeypatch.setattr(EmberContext, "get_config_path", staticmethod(lambda: prod_file))
        prod_ctx = EmberContext()

        assert prod_ctx.get_config("environment") == "production"
        assert prod_ctx.get_config("providers.openai.timeout") == 30
        assert prod_ctx.get_config("logging.level") == "ERROR"


class TestErrorHandling:
    """Test error handling in configuration system."""

    def test_malformed_config_handling(self, tmp_path, monkeypatch):
        """Test handling of malformed configurations."""
        config_file = tmp_path / ".ember" / "config.yaml"
        config_file.parent.mkdir(parents=True)

        # Write invalid YAML
        config_file.write_text("invalid: yaml: content: [")

        monkeypatch.setattr(EmberContext, "get_config_path", staticmethod(lambda: config_file))

        with pytest.raises(ValueError):
            EmberContext()

    def test_missing_required_fields(self, tmp_path, monkeypatch):
        """Test handling when required fields are missing."""
        config_file = tmp_path / ".ember" / "config.yaml"
        config_file.parent.mkdir(parents=True)

        # Config missing providers
        config = {
            "model": "gpt-4"
            # No providers section
        }

        config_file.write_text(yaml.dump(config))
        monkeypatch.setattr(EmberContext, "get_config_path", staticmethod(lambda: config_file))

        ctx = EmberContext()

        # Should handle missing providers gracefully
        assert ctx.get_config("providers", {}) == {}
        assert ctx.get_config("providers.openai", None) is None

    def test_circular_env_var_reference(self, tmp_path, monkeypatch):
        """Test handling of circular environment variable references."""
        # This shouldn't happen in practice, but test graceful handling
        config = {
            "providers": {"test": {"api_key": "${CIRCULAR_REF}", "other_key": "${CIRCULAR_REF}"}}
        }

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))

        # Load config - should not crash
        loaded = load_config(config_file)

        # Unresolved vars stay as placeholders
        assert loaded["providers"]["test"]["api_key"] == "${CIRCULAR_REF}"
