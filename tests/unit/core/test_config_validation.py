"""Configuration validation tests.

Each test validates exactly one aspect of configuration handling.
No magic, explicit assertions, fast execution.
"""

import pytest


class TestConfigKeyValidation:
    """Configuration key validation."""

    def test_string_key_required(self, tmp_ctx):
        """Non-string keys raise TypeError."""
        with pytest.raises(TypeError, match="Configuration key must be a string"):
            tmp_ctx.set_config(123, "value")

    def test_none_key_rejected(self, tmp_ctx):
        """None key raises TypeError."""
        with pytest.raises(TypeError, match="Configuration key must be a string"):
            tmp_ctx.set_config(None, "value")

    def test_empty_key_rejected(self, tmp_ctx):
        """Empty key raises ValueError."""
        with pytest.raises(ValueError, match="Configuration key cannot be empty"):
            tmp_ctx.set_config("", "value")

    def test_whitespace_key_rejected(self, tmp_ctx):
        """Whitespace-only key raises ValueError."""
        with pytest.raises(ValueError, match="Configuration key cannot be empty"):
            tmp_ctx.set_config("   ", "value")

    def test_leading_dot_rejected(self, tmp_ctx):
        """Key starting with dot raises ValueError."""
        with pytest.raises(ValueError, match="Invalid configuration key"):
            tmp_ctx.set_config(".key", "value")

    def test_trailing_dot_rejected(self, tmp_ctx):
        """Key ending with dot raises ValueError."""
        with pytest.raises(ValueError, match="Invalid configuration key"):
            tmp_ctx.set_config("key.", "value")

    def test_double_dot_rejected(self, tmp_ctx):
        """Key with double dots raises ValueError."""
        with pytest.raises(ValueError, match="Invalid configuration key"):
            tmp_ctx.set_config("key..subkey", "value")

    def test_space_in_key_accepted(self, tmp_ctx):
        """Key with space is now accepted for simplicity."""
        tmp_ctx.set_config("key with space", "value")
        assert tmp_ctx.get_config("key with space") == "value"

    def test_special_char_accepted(self, tmp_ctx):
        """Key with special character is now accepted for simplicity."""
        tmp_ctx.set_config("key@symbol", "value")
        assert tmp_ctx.get_config("key@symbol") == "value"


class TestConfigKeySuccess:
    """Valid configuration keys."""

    def test_simple_key_accepted(self, tmp_ctx):
        """Simple alphanumeric key works."""
        tmp_ctx.set_config("key", "value")
        assert tmp_ctx.get_config("key") == "value"

    def test_underscore_accepted(self, tmp_ctx):
        """Key with underscore works."""
        tmp_ctx.set_config("my_key", "value")
        assert tmp_ctx.get_config("my_key") == "value"

    def test_hyphen_accepted(self, tmp_ctx):
        """Key with hyphen works."""
        tmp_ctx.set_config("my-key", "value")
        assert tmp_ctx.get_config("my-key") == "value"

    def test_nested_key_accepted(self, tmp_ctx):
        """Dot-separated nested key works."""
        tmp_ctx.set_config("section.subsection.key", "value")
        assert tmp_ctx.get_config("section.subsection.key") == "value"

    def test_numeric_in_key_accepted(self, tmp_ctx):
        """Key with numbers works."""
        tmp_ctx.set_config("key123", "value")
        assert tmp_ctx.get_config("key123") == "value"


class TestConfigGetEdgeCases:
    """Edge cases for get_config."""

    @pytest.mark.parametrize("key", [None, "", 123, [], {}])
    def test_invalid_key_returns_none(self, tmp_ctx, key):
        """Invalid keys return None."""
        assert tmp_ctx.get_config(key) is None

    def test_missing_key_returns_none(self, tmp_ctx):
        """Missing key returns None."""
        assert tmp_ctx.get_config("no.such.key") is None

    def test_missing_key_returns_default(self, tmp_ctx):
        """Missing key returns provided default."""
        assert tmp_ctx.get_config("missing", "default") == "default"

    def test_partial_path_returns_dict(self, tmp_ctx):
        """Partial path returns sub-dictionary."""
        tmp_ctx.set_config("a.b.c", "value")
        result = tmp_ctx.get_config("a.b")
        assert isinstance(result, dict)
        assert result == {"c": "value"}
