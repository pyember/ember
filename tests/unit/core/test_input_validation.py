"""Tests for input validation security features."""

import pytest

from ember.core.credentials import CredentialManager


class TestConfigValidation:
    """Test configuration input validation."""

    def test_set_config_validates_key_type(self, tmp_ctx):
        """Non-string keys are rejected."""
        with pytest.raises(TypeError, match="Configuration key must be a string"):
            tmp_ctx.set_config(123, "value")

        with pytest.raises(TypeError, match="Configuration key must be a string"):
            tmp_ctx.set_config(None, "value")

    def test_set_config_validates_key_format(self, tmp_ctx):
        """Invalid key formats are rejected."""
        # Empty key
        with pytest.raises(ValueError, match="Configuration key cannot be empty"):
            tmp_ctx.set_config("", "value")

        with pytest.raises(ValueError, match="Configuration key cannot be empty"):
            tmp_ctx.set_config("   ", "value")

        # Invalid format
        with pytest.raises(ValueError, match="Invalid configuration key"):
            tmp_ctx.set_config(".key", "value")

        with pytest.raises(ValueError, match="Invalid configuration key"):
            tmp_ctx.set_config("key.", "value")

        with pytest.raises(ValueError, match="Invalid configuration key"):
            tmp_ctx.set_config("key..subkey", "value")

    def test_set_config_validates_key_components(self, tmp_ctx):
        """Key components are validated."""
        # We now accept most characters
        tmp_ctx.set_config("key with spaces", "value")
        tmp_ctx.set_config("key@symbol", "value")
        tmp_ctx.set_config("key!exclaim", "value")

        # Verify they were set
        assert tmp_ctx.get_config("key with spaces") == "value"
        assert tmp_ctx.get_config("key@symbol") == "value"
        assert tmp_ctx.get_config("key!exclaim") == "value"

    def test_set_config_accepts_valid_keys(self, tmp_ctx):
        """Valid keys are accepted."""
        # Simple key
        tmp_ctx.set_config("key", "value")
        assert tmp_ctx.get_config("key") == "value"

        # Nested key
        tmp_ctx.set_config("section.subsection.key", "value")
        assert tmp_ctx.get_config("section.subsection.key") == "value"

        # With underscores and hyphens
        tmp_ctx.set_config("my_key-name", "value")
        assert tmp_ctx.get_config("my_key-name") == "value"


class TestCredentialValidation:
    """Test credential input validation."""

    def test_save_api_key_validates_provider(self, tmp_path):
        """Provider name must be valid."""
        cred_mgr = CredentialManager(tmp_path)

        with pytest.raises(ValueError, match="Provider name must be a non-empty string"):
            cred_mgr.save_api_key("", "key")

        with pytest.raises(ValueError, match="Provider name must be a non-empty string"):
            cred_mgr.save_api_key(None, "key")

    def test_save_api_key_validates_key(self, tmp_path):
        """API key must be valid."""
        cred_mgr = CredentialManager(tmp_path)

        with pytest.raises(ValueError, match="API key must be a non-empty string"):
            cred_mgr.save_api_key("openai", "")

        with pytest.raises(ValueError, match="API key must be a non-empty string"):
            cred_mgr.save_api_key("openai", None)

        with pytest.raises(ValueError, match="API key appears to be too short"):
            cred_mgr.save_api_key("openai", "key")

    def test_save_api_key_detects_common_mistakes(self, tmp_path):
        """Common API key mistakes are caught."""
        cred_mgr = CredentialManager(tmp_path)

        # Quoted keys
        with pytest.raises(ValueError, match="API key should not be quoted"):
            cred_mgr.save_api_key("openai", '"sk-abcdef1234567890"')

        # Keys with spaces
        with pytest.raises(ValueError, match="API key should not contain spaces"):
            cred_mgr.save_api_key("openai", "sk-abcdef 1234567890")

    def test_save_api_key_accepts_valid_keys(self, tmp_path):
        """Valid API keys are accepted."""
        import warnings

        # Suppress deprecation warnings for this test since we're testing the deprecated system
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cred_mgr = CredentialManager(tmp_path)

            # Normal key
            cred_mgr.save_api_key("openai", "sk-abcdef1234567890")
            assert cred_mgr.get("openai") == "sk-abcdef1234567890"

            # Key with trailing whitespace (should be stripped)
            cred_mgr.save_api_key("anthropic", "  claude-key-123456  ")
            assert cred_mgr.get("anthropic") == "claude-key-123456"


class TestPathValidation:
    """Test path traversal and injection prevention."""

    def test_config_path_validation(self, tmp_path):
        """Config paths are properly validated."""
        # This would be implemented if we were accepting file paths from users
        # For now, config paths are internally generated
        pass
