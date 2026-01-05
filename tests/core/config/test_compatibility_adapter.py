"""Tests for configuration compatibility adapter."""

from ember.core.config.compatibility_adapter import CompatibilityAdapter


class TestExternalFormatDetection:
    """Test external format detection."""

    def test_detect_external_by_top_level_fields(self):
        """Test detection by external tool-specific top-level fields."""
        config = {"model": "o4-mini", "approvalMode": "suggest", "providers": {}}
        assert CompatibilityAdapter.needs_adaptation(config) is True

        config = {"model": "gpt-4", "fullAutoErrorMode": "ask-user", "providers": {}}
        assert CompatibilityAdapter.needs_adaptation(config) is True

        config = {"model": "gpt-4", "notify": True, "providers": {}}
        assert CompatibilityAdapter.needs_adaptation(config) is True

    def test_detect_external_by_provider_format(self):
        """Test detection by provider configuration format."""
        config = {
            "providers": {
                "openai": {
                    "name": "OpenAI",
                    "baseURL": "https://api.openai.com/v1",
                    "envKey": "OPENAI_API_KEY",
                    # No api_key field
                }
            }
        }
        assert CompatibilityAdapter.needs_adaptation(config) is True

    def test_not_external_format(self):
        """Test non-external format detection."""
        # Ember format with api_key
        config = {
            "providers": {"openai": {"api_key": "sk-123", "base_url": "https://api.openai.com/v1"}}
        }
        assert CompatibilityAdapter.needs_adaptation(config) is False

        # Empty config
        assert CompatibilityAdapter.needs_adaptation({}) is False

        # No providers
        config = {"model": "gpt-4", "temperature": 0.7}
        assert CompatibilityAdapter.needs_adaptation(config) is False


class TestProviderAdaptation:
    """Test provider configuration adaptation."""

    def test_adapt_provider_with_env_key(self, monkeypatch):
        """Test adapting provider with envKey."""
        external_provider = {
            "name": "OpenAI",
            "baseURL": "https://api.openai.com/v1",
            "envKey": "OPENAI_API_KEY",
        }

        adapted = CompatibilityAdapter.adapt_provider_config(external_provider)

        assert adapted["name"] == "OpenAI"
        assert adapted["baseURL"] == "https://api.openai.com/v1"
        assert adapted["base_url"] == "https://api.openai.com/v1"  # Mapped field
        assert adapted["envKey"] == "OPENAI_API_KEY"
        assert adapted["env_key"] == "OPENAI_API_KEY"
        assert "api_key" not in adapted

    def test_adapt_provider_missing_env(self):
        """Test adapting provider that only specifies envKey."""
        external_provider = {
            "name": "OpenAI",
            "baseURL": "https://api.openai.com/v1",
            "envKey": "MISSING_API_KEY",
        }

        adapted = CompatibilityAdapter.adapt_provider_config(external_provider)

        assert adapted["env_key"] == "MISSING_API_KEY"
        assert "api_key" not in adapted

    def test_adapt_provider_already_has_api_key(self):
        """Test adapting provider that already has api_key."""
        provider = {
            "name": "OpenAI",
            "baseURL": "https://api.openai.com/v1",
            "api_key": "existing-key",
        }

        adapted = CompatibilityAdapter.adapt_provider_config(provider)

        assert adapted["api_key"] == "existing-key"  # Unchanged
        assert "envKey" not in adapted  # No envKey added

    def test_adapt_all_external_providers(self, monkeypatch):
        """Test adapting all known external providers."""
        providers = {
            "openai": {
                "name": "OpenAI",
                "baseURL": "https://api.openai.com/v1",
                "envKey": "OPENAI_API_KEY",
            },
            "azure": {
                "name": "AzureOpenAI",
                "baseURL": "https://project.openai.azure.com/openai",
                "envKey": "AZURE_OPENAI_API_KEY",
            },
            "openrouter": {
                "name": "OpenRouter",
                "baseURL": "https://openrouter.ai/api/v1",
                "envKey": "OPENROUTER_API_KEY",
            },
            "gemini": {
                "name": "Gemini",
                "baseURL": "https://generativelanguage.googleapis.com/v1beta/openai",
                "envKey": "GOOGLE_API_KEY",
            },
            "ollama": {
                "name": "Ollama",
                "baseURL": "http://localhost:11434/v1",
                "envKey": "OLLAMA_API_KEY",
            },
            "mistral": {
                "name": "Mistral",
                "baseURL": "https://api.mistral.ai/v1",
                "envKey": "MISTRAL_API_KEY",
            },
            "deepseek": {
                "name": "DeepSeek",
                "baseURL": "https://api.deepseek.com",
                "envKey": "DEEPSEEK_API_KEY",
            },
            "xai": {
                "name": "xAI",
                "baseURL": "https://api.x.ai/v1",
                "envKey": "XAI_API_KEY",
            },
            "groq": {
                "name": "Groq",
                "baseURL": "https://api.groq.com/openai/v1",
                "envKey": "GROQ_API_KEY",
            },
            "arceeai": {
                "name": "ArceeAI",
                "baseURL": "https://conductor.arcee.ai/v1",
                "envKey": "ARCEEAI_API_KEY",
            },
        }

        for _provider_name, provider_config in providers.items():
            adapted = CompatibilityAdapter.adapt_provider_config(provider_config)

            assert adapted["env_key"] == provider_config["envKey"]
            assert "api_key" not in adapted

            # Check baseURL mapping
            assert adapted["base_url"] == provider_config["baseURL"]


class TestConfigAdaptation:
    """Test full configuration adaptation."""

    def test_adapt_full_external_config(self, monkeypatch):
        """Test adapting complete external configuration."""
        external_config = {
            "model": "o4-mini",
            "provider": "openai",
            "approvalMode": "suggest",
            "fullAutoErrorMode": "ask-user",
            "notify": True,
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
            "history": {"maxSize": 1000, "saveHistory": True},
        }

        adapted = CompatibilityAdapter.adapt_config(external_config)

        # Check providers were adapted
        assert adapted["providers"]["openai"]["env_key"] == "OPENAI_API_KEY"
        assert adapted["providers"]["anthropic"]["env_key"] == "ANTHROPIC_API_KEY"

        # Check Codex fields were preserved
        assert adapted["_external_compat"]["approvalMode"] == "suggest"
        assert adapted["_external_compat"]["fullAutoErrorMode"] == "ask-user"
        assert adapted["_external_compat"]["notify"] is True

        # Check other fields remain
        assert adapted["model"] == "o4-mini"
        assert adapted["provider"] == "openai"
        assert adapted["history"]["maxSize"] == 1000

    def test_adapt_non_external_config(self):
        """Test that non-external configs are returned unchanged."""
        ember_config = {
            "version": "1.0",
            "providers": {
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",
                    "base_url": "https://api.openai.com/v1",
                }
            },
        }

        adapted = CompatibilityAdapter.adapt_config(ember_config)

        # Should be unchanged
        assert adapted == ember_config
        assert "_external_compat" not in adapted

    def test_adapt_empty_config(self):
        """Test adapting empty configuration."""
        adapted = CompatibilityAdapter.adapt_config({})
        assert adapted == {}


class TestProviderMigration:
    """Test explicit provider migration."""

    def test_migrate_external_provider(self):
        """Test migrating external provider for explicit migration."""
        external_provider = {
            "name": "OpenAI",
            "baseURL": "https://api.openai.com/v1",
            "envKey": "OPENAI_API_KEY",
            "extraField": "value",
        }

        migrated = CompatibilityAdapter.migrate_provider(external_provider)

        assert migrated["name"] == "OpenAI"
        assert migrated["base_url"] == "https://api.openai.com/v1"
        assert migrated["env_key"] == "OPENAI_API_KEY"
        assert migrated["_original"] == external_provider
        assert "api_key" not in migrated

    def test_migrate_minimal_provider(self):
        """Test migrating provider with minimal fields."""
        external_provider = {"envKey": "API_KEY"}

        migrated = CompatibilityAdapter.migrate_provider(external_provider)

        assert migrated["name"] == ""
        assert migrated["base_url"] == ""
        assert migrated["env_key"] == "API_KEY"
        assert migrated["_original"] == external_provider
        assert "api_key" not in migrated


class TestRealWorldScenarios:
    """Test real-world external configuration scenarios."""

    def test_direct_paste_external_config(self, monkeypatch):
        """Test user directly pasting external config."""
        # Exact external config from documentation
        external_config = {
            "model": "o4-mini",
            "approvalMode": "suggest",
            "fullAutoErrorMode": "ask-user",
            "notify": True,
            "provider": "openai",
            "providers": {
                "openai": {
                    "name": "OpenAI",
                    "baseURL": "https://api.openai.com/v1",
                    "envKey": "OPENAI_API_KEY",
                }
            },
        }

        # User copies this to ~/.ember/config.json
        adapted = CompatibilityAdapter.adapt_config(external_config)

        assert adapted["providers"]["openai"]["env_key"] == "OPENAI_API_KEY"
        assert adapted["model"] == "o4-mini"
        assert adapted["provider"] == "openai"

    def test_partial_external_config(self):
        """Test partial external configuration."""
        config = {
            "model": "gpt-4",
            "approvalMode": "auto-edit",
            # No providers section
        }

        adapted = CompatibilityAdapter.adapt_config(config)

        assert adapted["_external_compat"]["approvalMode"] == "auto-edit"
        assert adapted["model"] == "gpt-4"
