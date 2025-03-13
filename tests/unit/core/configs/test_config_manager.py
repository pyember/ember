"""Tests for the ConfigManager class.

This module tests the thread-safe configuration management system
including loading, transformation, validation, and access.
"""

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any
from unittest import mock

import pytest
from pydantic import BaseModel, ValidationError

from src.ember.core.configs.config_manager import ConfigManager
from src.ember.core.configs.providers import ConfigProvider, YamlFileProvider, EnvironmentProvider
from src.ember.core.configs.transformer import ConfigTransformer
from src.ember.core.configs.exceptions import ConfigError
from src.ember.core.configs.schema import EmberConfig


class SimpleConfig(BaseModel):
    """Simple configuration model for testing."""
    name: str
    value: int
    nested: Dict[str, Any] = {}


class MockProvider(ConfigProvider):
    """Mock config provider for testing."""
    
    def __init__(self, config_data=None, save_error=False):
        self.config_data = config_data or {}
        self.save_error = save_error
        self.save_called = False
        self.load_called = False
    
    def load(self):
        self.load_called = True
        return self.config_data
    
    def save(self, config):
        self.save_called = True
        self.config_data = config
        if self.save_error:
            raise Exception("Mock save error")


class MockTransformer(ConfigTransformer):
    """Mock transformer for testing."""
    
    def __init__(self, transform_func=None):
        super().__init__()
        self.transform_called = False
        self.transform_func = transform_func
    
    def transform(self, config):
        self.transform_called = True
        if self.transform_func:
            return self.transform_func(config)
        return config


class TestConfigManager:
    """Test suite for ConfigManager class."""

    def test_initialization(self):
        """Test ConfigManager initialization."""
        provider = MockProvider()
        transformer = MockTransformer()
        manager = ConfigManager(SimpleConfig, [provider], [transformer])
        
        assert manager._schema_class == SimpleConfig
        assert manager._providers == [provider]
        assert manager._transformers == [transformer]
        assert manager._config is None

    def test_load_basic(self):
        """Test basic configuration loading."""
        provider = MockProvider({"name": "test", "value": 42})
        manager = ConfigManager(SimpleConfig, [provider])
        
        config = manager.load()
        
        assert provider.load_called
        assert isinstance(config, SimpleConfig)
        assert config.name == "test"
        assert config.value == 42

    def test_load_with_transformers(self):
        """Test configuration loading with transformers."""
        provider = MockProvider({"name": "test", "value": 42})
        
        def transform_func(config):
            config["value"] *= 2
            return config
            
        transformer = MockTransformer(transform_func)
        manager = ConfigManager(SimpleConfig, [provider], [transformer])
        
        config = manager.load()
        
        assert provider.load_called
        assert transformer.transform_called
        assert config.value == 84

    def test_multiple_providers(self):
        """Test loading from multiple providers with overrides."""
        provider1 = MockProvider({"name": "test1", "value": 10})
        provider2 = MockProvider({"name": "test2", "value": 20})
        manager = ConfigManager(SimpleConfig, [provider1, provider2])
        
        config = manager.load()
        
        # Later providers should override earlier ones
        assert config.name == "test2"
        assert config.value == 20

    def test_validation_error(self):
        """Test handling of validation errors."""
        # Missing required field 'value'
        provider = MockProvider({"name": "test"})
        manager = ConfigManager(SimpleConfig, [provider])
        
        with pytest.raises(ConfigError) as exc_info:
            manager.load()
            
        assert "validation" in str(exc_info.value).lower()

    def test_provider_error(self):
        """Test handling of provider errors."""
        provider = MockProvider()
        provider.load = mock.MagicMock(side_effect=Exception("Provider error"))
        manager = ConfigManager(SimpleConfig, [provider, MockProvider({"name": "fallback", "value": 99})])
        
        config = manager.load()
        
        # Should continue to next provider
        assert config.name == "fallback"
        assert config.value == 99

    def test_get_config(self):
        """Test getting configuration."""
        provider = MockProvider({"name": "test", "value": 42})
        manager = ConfigManager(SimpleConfig, [provider])
        
        # First call loads config
        config1 = manager.get_config()
        assert provider.load_called
        assert config1.name == "test"
        
        # Reset mock to check if loaded again
        provider.load_called = False
        
        # Second call should not reload
        config2 = manager.get_config()
        assert not provider.load_called
        assert config2 is config1  # Should be the same instance

    def test_reload(self):
        """Test configuration reloading."""
        provider = MockProvider({"name": "test", "value": 42})
        manager = ConfigManager(SimpleConfig, [provider])
        
        # First load
        config1 = manager.get_config()
        assert config1.name == "test"
        assert config1.value == 42
        
        # Change provider data
        provider.config_data = {"name": "updated", "value": 100}
        
        # Reload should get new data
        config2 = manager.reload()
        assert config2.name == "updated"
        assert config2.value == 100
        assert config2 is not config1  # Should be a new instance

    def test_save(self):
        """Test saving configuration."""
        provider = MockProvider({"name": "test", "value": 42})
        manager = ConfigManager(SimpleConfig, [provider])
        
        # Load and modify config
        config = manager.get_config()
        
        # Simulate modified config
        manager._config = SimpleConfig(name="modified", value=99)
        
        # Save back to provider
        manager.save()
        
        assert provider.save_called
        assert provider.config_data["name"] == "modified"
        assert provider.config_data["value"] == 99

    def test_save_error(self):
        """Test handling of save errors."""
        provider = MockProvider({"name": "test", "value": 42}, save_error=True)
        manager = ConfigManager(SimpleConfig, [provider])
        
        manager.get_config()  # Load config
        
        with pytest.raises(ConfigError) as exc_info:
            manager.save()
            
        assert "failed to save" in str(exc_info.value).lower()

    def test_get_path(self):
        """Test getting config values by path."""
        provider = MockProvider({
            "name": "test", 
            "value": 42,
            "nested": {
                "level1": {
                    "level2": "deep value"
                }
            }
        })
        manager = ConfigManager(SimpleConfig, [provider])
        
        assert manager.get("name") == "test"
        assert manager.get("value") == 42
        assert manager.get("nested", "level1", "level2") == "deep value"
        assert manager.get("missing") is None
        assert manager.get("nested", "missing", default="fallback") == "fallback"

    def test_thread_safety(self):
        """Test thread safety with concurrent access."""
        provider = MockProvider({"name": "test", "value": 42})
        manager = ConfigManager(SimpleConfig, [provider])
        
        # Slow transformer to increase chance of race conditions
        def slow_transform(config):
            time.sleep(0.1)
            config["thread_id"] = threading.get_ident()
            return config
            
        manager._transformers = [MockTransformer(slow_transform)]
        
        results = {}
        errors = []
        
        def worker(idx):
            try:
                config = manager.get_config()
                results[idx] = config.model_dump()
            except Exception as e:
                errors.append(e)
        
        # Create threads
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        
        # Start threads
        for t in threads:
            t.start()
            
        # Wait for threads to complete
        for t in threads:
            t.join()
            
        assert not errors, f"Errors occurred: {errors}"
        
        # All threads should see the same configuration 
        first_thread_id = results[0]["thread_id"]
        for idx, result in results.items():
            assert result["thread_id"] == first_thread_id

    def test_set_provider_api_key(self):
        """Test setting provider API key."""
        initial_config = {
            "model_registry": {
                "providers": {
                    "existing_provider": {
                        "enabled": True
                    }
                }
            }
        }
        
        provider = MockProvider(initial_config)
        manager = ConfigManager(EmberConfig, [provider])
        
        # Add API key to existing provider
        manager.set_provider_api_key("existing_provider", "test_api_key_1")
        
        # Check config was updated
        config = manager.get_config()
        assert config.model_registry.providers["existing_provider"].api_keys["default"].key == "test_api_key_1"
        
        # Add API key to new provider
        manager.set_provider_api_key("new_provider", "test_api_key_2")
        
        # Check new provider was added with API key
        config = manager.get_config()
        assert config.model_registry.providers["new_provider"].enabled is True
        assert config.model_registry.providers["new_provider"].api_keys["default"].key == "test_api_key_2"
        
        # Verify save was called
        assert provider.save_called

    @pytest.mark.parametrize("test_input,expected", [
        ({"name": "test", "value": 42}, SimpleConfig(name="test", value=42)),
        ({"name": "test", "value": "42"}, SimpleConfig(name="test", value=42))  # Type conversion
    ])
    def test_schema_validation(self, test_input, expected):
        """Test schema validation with various inputs."""
        provider = MockProvider(test_input)
        manager = ConfigManager(SimpleConfig, [provider])
        
        config = manager.load()
        
        assert config.name == expected.name
        assert config.value == expected.value


class TestYamlFileProvider:
    """Test suite for YamlFileProvider."""
    
    def test_yaml_load(self):
        """Test loading from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
            f.write("""
            name: yaml_test
            value: 100
            nested:
              key: value
            """)
        
        try:
            provider = YamlFileProvider(yaml_path)
            config = provider.load()
            
            assert config["name"] == "yaml_test"
            assert config["value"] == 100
            assert config["nested"]["key"] == "value"
        finally:
            os.unlink(yaml_path)
            
    def test_missing_file(self):
        """Test behavior with missing file."""
        provider = YamlFileProvider("/not/a/real/path.yaml")
        
        # Should return empty dict, not error
        config = provider.load()
        assert config == {}
        
    def test_yaml_save(self):
        """Test saving to YAML file."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            provider = YamlFileProvider(yaml_path)
            config_data = {
                "name": "save_test",
                "value": 200,
                "nested": {"saved": True}
            }
            
            provider.save(config_data)
            
            # Verify by loading again
            loaded = provider.load()
            assert loaded["name"] == "save_test"
            assert loaded["value"] == 200
            assert loaded["nested"]["saved"] is True
        finally:
            os.unlink(yaml_path)


class TestEnvironmentProvider:
    """Test suite for EnvironmentProvider."""
    
    def test_env_load(self):
        """Test loading from environment variables."""
        # Set test environment variables
        with mock.patch.dict(os.environ, {
            "TEST_NAME": "env_test",
            "TEST_VALUE": "42",
            "TEST_NESTED__KEY": "nested_value"
        }):
            provider = EnvironmentProvider(prefix="TEST")
            config = provider.load()
            
            assert config["name"] == "env_test"
            assert config["value"] == "42"  # Note: still a string, transformers handle conversion
            assert config["nested"]["key"] == "nested_value"
    
    def test_empty_environment(self):
        """Test with no matching environment variables."""
        with mock.patch.dict(os.environ, clear=True):
            provider = EnvironmentProvider(prefix="TEST")
            config = provider.load()
            
            assert config == {}
    
    def test_save_not_implemented(self):
        """Test that save is not implemented for environment variables."""
        provider = EnvironmentProvider(prefix="TEST")
        
        with pytest.raises(NotImplementedError):
            provider.save({"name": "test"})