"""Tests for the configuration loader module.

This module contains tests for the configuration loading functionality in ember.core.config.loader.
"""

import os
import tempfile

import pytest
import yaml

from ember.core.config.exceptions import ConfigError
from ember.core.config.loader import (
    load_config,
    load_from_env,
    load_yaml_file,
    merge_dicts,
    resolve_env_vars,
)
from ember.core.config.schema import EmberConfig


class TestMergeDicts:
    """Tests for the merge_dicts function."""

    def test_simple_merge(self):
        """Test merging simple dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_dicts(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Test merging nested dictionaries."""
        base = {"a": 1, "b": {"x": 1, "y": 2}}
        override = {"b": {"y": 3, "z": 4}, "c": 5}
        result = merge_dicts(base, override)
        assert result == {"a": 1, "b": {"x": 1, "y": 3, "z": 4}, "c": 5}

    def test_empty_dicts(self):
        """Test merging with empty dictionaries."""
        base = {}
        override = {"a": 1}
        result = merge_dicts(base, override)
        assert result == {"a": 1}

        base = {"a": 1}
        override = {}
        result = merge_dicts(base, override)
        assert result == {"a": 1}

    def test_original_unchanged(self):
        """Test that original dictionaries are not modified."""
        base = {"a": 1}
        override = {"b": 2}
        result = merge_dicts(base, override)
        assert base == {"a": 1}
        assert override == {"b": 2}


class TestResolveEnvVars:
    """Tests for the resolve_env_vars function."""

    def test_simple_resolve(self, monkeypatch):
        """Test resolving simple environment variables."""
        monkeypatch.setenv("TEST_VAR", "test-value")
        config = {"key": "${TEST_VAR}"}
        result = resolve_env_vars(config)
        assert result == {"key": "test-value"}

    def test_partial_resolve(self, monkeypatch):
        """Test resolving environment variables within strings."""
        monkeypatch.setenv("TEST_VAR", "test-value")
        config = {"key": "prefix-${TEST_VAR}-suffix"}
        result = resolve_env_vars(config)
        assert result == {"key": "prefix-test-value-suffix"}

    def test_multiple_resolve(self, monkeypatch):
        """Test resolving multiple environment variables."""
        monkeypatch.setenv("TEST_VAR1", "value1")
        monkeypatch.setenv("TEST_VAR2", "value2")
        config = {"key1": "${TEST_VAR1}", "key2": "${TEST_VAR2}"}
        result = resolve_env_vars(config)
        assert result == {"key1": "value1", "key2": "value2"}

    def test_nested_resolve(self, monkeypatch):
        """Test resolving environment variables in nested structures."""
        monkeypatch.setenv("TEST_VAR", "test-value")
        config = {"outer": {"inner": "${TEST_VAR}"}}
        result = resolve_env_vars(config)
        assert result == {"outer": {"inner": "test-value"}}

    def test_list_resolve(self, monkeypatch):
        """Test resolving environment variables in lists."""
        monkeypatch.setenv("TEST_VAR", "test-value")
        config = {"list": ["item1", "${TEST_VAR}", "item3"]}
        result = resolve_env_vars(config)
        assert result == {"list": ["item1", "test-value", "item3"]}

    def test_missing_var(self, monkeypatch):
        """Test resolving non-existent environment variables."""
        # Ensure environment variable doesn't exist
        if "NON_EXISTENT_VAR" in os.environ:
            monkeypatch.delenv("NON_EXISTENT_VAR")

        config = {"key": "${NON_EXISTENT_VAR}"}
        result = resolve_env_vars(config)
        assert result == {"key": ""}

    def test_non_dict_input(self):
        """Test resolving with non-dictionary input."""
        result = resolve_env_vars("not-a-dict")
        assert result == "not-a-dict"


class TestLoadYamlFile:
    """Tests for the load_yaml_file function."""

    def test_valid_yaml(self):
        """Test loading a valid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump({"key": "value"}, f)
            f.flush()
            result = load_yaml_file(f.name)
            assert result == {"key": "value"}

    def test_invalid_yaml(self):
        """Test loading an invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            f.write("invalid: yaml: :")
            f.flush()
            with pytest.raises(ConfigError):
                load_yaml_file(f.name)

    def test_non_existent_file(self):
        """Test loading a non-existent file."""
        result = load_yaml_file("/non/existent/file.yaml")
        assert result == {}

    def test_empty_file(self):
        """Test loading an empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            result = load_yaml_file(f.name)
            assert result == {}


class TestLoadFromEnv:
    """Tests for the load_from_env function."""

    def test_simple_env(self, monkeypatch):
        """Test loading simple environment variables."""
        monkeypatch.setenv("EMBER_KEY", "value")
        result = load_from_env()
        assert result == {"key": "value"}

    def test_nested_env(self, monkeypatch):
        """Test loading nested environment variables."""
        monkeypatch.setenv("EMBER_OUTER_INNER", "value")
        result = load_from_env()
        assert result == {"outer": {"inner": "value"}}

    def test_multiple_env(self, monkeypatch):
        """Test loading multiple environment variables."""
        monkeypatch.setenv("EMBER_KEY1", "value1")
        monkeypatch.setenv("EMBER_KEY2", "value2")
        result = load_from_env()
        assert result == {"key1": "value1", "key2": "value2"}

    def test_custom_prefix(self, monkeypatch):
        """Test loading with custom prefix."""
        monkeypatch.setenv("CUSTOM_KEY", "value")
        result = load_from_env(prefix="CUSTOM")
        assert result == {"key": "value"}

    def test_type_conversion(self, monkeypatch):
        """Test type conversion of environment variables."""
        monkeypatch.setenv("EMBER_INT", "42")
        monkeypatch.setenv("EMBER_FLOAT", "3.14")
        monkeypatch.setenv("EMBER_TRUE", "true")
        monkeypatch.setenv("EMBER_FALSE", "false")
        monkeypatch.setenv("EMBER_STRING", "string")

        result = load_from_env()
        assert result == {
            "int": 42,
            "float": 3.14,
            "true": True,
            "false": False,
            "string": "string",
        }

    def test_no_matching_env(self, monkeypatch):
        """Test with no matching environment variables."""
        # Ensure no EMBER_ variables exist
        for key in list(os.environ.keys()):
            if key.startswith("EMBER_"):
                monkeypatch.delenv(key)

        result = load_from_env()
        assert result == {}


@pytest.fixture
def config_file():
    """Create a temporary configuration file."""
    config = {
        "registry": {
            "auto_discover": True,
            "providers": {
                "provider1": {
                    "enabled": True,
                    "api_key": "key1",
                    "models": {
                        "model1": {
                            "id": "model1",
                            "name": "Model One",
                            "provider": "provider1",
                            "cost_input": 1.0,
                            "cost_output": 2.0,
                        }
                    },
                }
            },
        },
        "logging": {"level": "INFO"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        yaml.dump(config, f)
        f.flush()
        yield f.name


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_from_file(self, config_file):
        """Test loading configuration from a file."""
        config = load_config(file_path=config_file)

        assert isinstance(config, EmberConfig)
        assert config.registry.auto_discover is True
        assert len(config.registry.providers) == 1
        assert config.registry.providers["provider1"].enabled is True
        assert config.registry.providers["provider1"].api_key == "key1"
        assert len(config.registry.providers["provider1"].models) == 1
        assert (
            config.registry.providers["provider1"].models["model1"].name == "Model One"
        )
        assert config.logging.level == "INFO"

    def test_load_with_env_override(self, config_file, monkeypatch):
        """Test loading with environment variable overrides."""
        monkeypatch.setenv("EMBER_REGISTRY_AUTO_DISCOVER", "false")
        monkeypatch.setenv("EMBER_LOGGING_LEVEL", "DEBUG")

        config = load_config(file_path=config_file)

        assert config.registry.auto_discover is False  # Overridden by env
        assert config.logging.level == "DEBUG"  # Overridden by env
        assert config.registry.providers["provider1"].api_key == "key1"  # From file

    def test_load_with_env_var_substitution(self, monkeypatch):
        """Test loading with environment variable substitution."""
        config = {
            "registry": {"providers": {"provider1": {"api_key": "${TEST_API_KEY}"}}}
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump(config, f)
            f.flush()

            monkeypatch.setenv("TEST_API_KEY", "test-api-key")

            config = load_config(file_path=f.name)

            assert config.registry.providers["provider1"].api_key == "test-api-key"

    def test_invalid_config(self):
        """Test loading an invalid configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            f.write("invalid: yaml: :")
            f.flush()

            with pytest.raises(ConfigError):
                load_config(file_path=f.name)

    def test_non_existent_file(self):
        """Test loading from a non-existent file."""
        config = load_config(file_path="/non/existent/file.yaml")

        # Should still return a valid config with defaults
        assert isinstance(config, EmberConfig)
        assert config.registry.auto_discover is True

    def test_custom_env_prefix(self, monkeypatch):
        """Test loading with custom environment prefix."""
        monkeypatch.setenv("CUSTOM_REGISTRY_AUTO_DISCOVER", "false")

        config = load_config(env_prefix="CUSTOM")

        # Check if the auto entry exists within the registry config
        if hasattr(config.registry, "auto") and isinstance(config.registry.auto, dict):
            assert config.registry.auto.get("discover") is False
        else:
            assert config.registry.auto_discover is False
