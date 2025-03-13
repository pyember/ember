"""Tests for the ConfigProvider interface contracts.

This module focuses on testing that the ConfigProvider interface and
its implementations adhere to SOLID principles, particularly the
Liskov Substitution Principle and Interface Segregation.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Protocol, runtime_checkable
from unittest import mock

import pytest
from pydantic import BaseModel

from src.ember.core.configs.providers import ConfigProvider, YamlFileProvider, EnvironmentProvider


@runtime_checkable
class MinimalProviderProtocol(Protocol):
    """A minimal protocol that all providers should satisfy."""
    
    def load(self) -> Dict[str, Any]:
        """Load configuration data from the source."""
        ...


class TestProviderContracts:
    """Tests for the ConfigProvider interface contracts."""

    def test_all_providers_satisfy_minimal_protocol(self):
        """Test that all providers satisfy the minimal provider protocol."""
        # Create instances of all provider implementations
        providers = [
            YamlFileProvider("dummy.yaml"),
            EnvironmentProvider(prefix="TEST")
        ]
        
        # Check that each provider satisfies the protocol
        for provider in providers:
            assert isinstance(provider, MinimalProviderProtocol), f"{provider.__class__.__name__} does not satisfy MinimalProviderProtocol"
    
    def test_provider_base_class_contract(self):
        """Test that the base ConfigProvider class defines the correct interface."""
        # The base class should define load and save methods
        assert hasattr(ConfigProvider, "load")
        assert hasattr(ConfigProvider, "save")
        
        # Create a minimal subclass
        class MinimalProvider(ConfigProvider):
            def load(self):
                return {}
        
        # Should be able to instantiate without implementing save
        # (save can have a default implementation or be optional)
        provider = MinimalProvider()
        assert provider.load() == {}
        
        # The save method might raise NotImplementedError, which is acceptable
        # for providers that don't support saving
        try:
            provider.save({})
        except NotImplementedError:
            pass  # This is acceptable behavior
    
    def test_provider_implementation_independence(self):
        """Test that provider implementations are independent of each other."""
        # This test ensures that changes to one provider implementation
        # don't affect others (Open/Closed Principle)
        
        # First, verify normal behavior of YamlFileProvider
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
            f.write("key: value")
        
        try:
            yaml_provider = YamlFileProvider(yaml_path)
            yaml_result = yaml_provider.load()
            assert yaml_result == {"key": "value"}
            
            # Now mock the load method of YamlFileProvider to return something different
            original_yaml_load = YamlFileProvider.load
            YamlFileProvider.load = mock.MagicMock(return_value={"mocked": True})
            
            # This change should not affect EnvironmentProvider
            with mock.patch.dict(os.environ, {"TEST_ENV_KEY": "env_value"}):
                env_provider = EnvironmentProvider(prefix="TEST")
                env_result = env_provider.load()
                
                # The environment provider should still work normally
                assert env_result == {"env_key": "env_value"}
                
                # But the YAML provider now returns the mocked value
                assert YamlFileProvider(yaml_path).load() == {"mocked": True}
        finally:
            # Restore the original method
            if 'original_yaml_load' in locals():
                YamlFileProvider.load = original_yaml_load
            os.unlink(yaml_path)
    
    def test_liskov_substitution_principle(self):
        """Test that providers can be substituted for one another."""
        # This test verifies that any provider can be used where a
        # ConfigProvider is expected (Liskov Substitution Principle)
        
        # Create a function that takes a ConfigProvider
        def process_config(provider: ConfigProvider) -> Dict[str, Any]:
            config = provider.load()
            # Add a value to the config
            config["processed"] = True
            return config
        
        # Test with YamlFileProvider
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
            f.write("yaml_key: yaml_value")
        
        try:
            yaml_provider = YamlFileProvider(yaml_path)
            yaml_result = process_config(yaml_provider)
            assert yaml_result == {"yaml_key": "yaml_value", "processed": True}
            
            # Test with EnvironmentProvider
            with mock.patch.dict(os.environ, {"TEST_ENV_KEY": "env_value"}):
                env_provider = EnvironmentProvider(prefix="TEST")
                env_result = process_config(env_provider)
                assert env_result == {"env_key": "env_value", "processed": True}
                
                # Test with a custom provider implementation
                class CustomProvider(ConfigProvider):
                    def load(self):
                        return {"custom_key": "custom_value"}
                    
                    def save(self, config):
                        pass  # Do nothing
                
                custom_provider = CustomProvider()
                custom_result = process_config(custom_provider)
                assert custom_result == {"custom_key": "custom_value", "processed": True}
        finally:
            os.unlink(yaml_path)
    
    def test_single_responsibility_principle(self):
        """Test that each provider has a single responsibility."""
        # Each provider should be focused on a single source of configuration
        
        # YamlFileProvider should only load from YAML files
        yaml_provider = YamlFileProvider("config.yaml")
        
        # Mock file existence and content
        with mock.patch("pathlib.Path.exists", return_value=True):
            with mock.patch("builtins.open", mock.mock_open(read_data="key: value")):
                yaml_config = yaml_provider.load()
                assert yaml_config == {"key": "value"}
        
        # Changing environment variables should not affect YamlFileProvider
        with mock.patch.dict(os.environ, {"YAML_KEY": "new_value"}):
            with mock.patch("pathlib.Path.exists", return_value=True):
                with mock.patch("builtins.open", mock.mock_open(read_data="key: value")):
                    yaml_config = yaml_provider.load()
                    # Still the same, not affected by env vars
                    assert yaml_config == {"key": "value"}
        
        # EnvironmentProvider should only load from environment variables
        env_provider = EnvironmentProvider(prefix="TEST")
        
        # Set an environment variable
        with mock.patch.dict(os.environ, {"TEST_ENV_KEY": "env_value"}):
            env_config = env_provider.load()
            assert env_config == {"env_key": "env_value"}
            
            # Creating a YAML file should not affect EnvironmentProvider
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
                f.write("yaml_key: yaml_value")
                f.flush()
                
                # Should still only read from environment
                env_config = env_provider.load()
                assert env_config == {"env_key": "env_value"}
    
    def test_interface_segregation_principle(self):
        """Test that provider interfaces are minimal and focused."""
        # Providers should only depend on methods they actually use
        
        # Create a minimal read-only provider that only implements load
        class ReadOnlyProvider(ConfigProvider):
            def load(self):
                return {"read_only": True}
        
        read_only_provider = ReadOnlyProvider()
        
        # Should be able to use the provider without calling save
        assert read_only_provider.load() == {"read_only": True}
        
        # Create a read-write provider that implements both load and save
        class ReadWriteProvider(ConfigProvider):
            def __init__(self):
                self.data = {"read_write": True}
            
            def load(self):
                return self.data
            
            def save(self, config):
                self.data = config
        
        read_write_provider = ReadWriteProvider()
        
        # Should be able to use both methods
        assert read_write_provider.load() == {"read_write": True}
        read_write_provider.save({"updated": True})
        assert read_write_provider.load() == {"updated": True}


class TestYamlFileProvider:
    """Tests for the YamlFileProvider implementation."""
    
    def test_yaml_provider_contract(self):
        """Test that YamlFileProvider implements the contract correctly."""
        provider = YamlFileProvider("config.yaml")
        
        # Should implement both load and save
        assert hasattr(provider, "load")
        assert hasattr(provider, "save")
        
        # Path should be stored and accessible
        assert provider.file_path == "config.yaml"
    
    def test_yaml_provider_error_handling(self):
        """Test that YamlFileProvider handles errors gracefully."""
        # Should handle missing files by returning empty dict
        provider = YamlFileProvider("/non/existent/path.yaml")
        assert provider.load() == {}
        
        # Should handle invalid YAML gracefully
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
            f.write("invalid: yaml: content:")  # Invalid YAML
        
        try:
            provider = YamlFileProvider(yaml_path)
            
            # Should raise an appropriate error
            with pytest.raises(Exception) as exc_info:
                provider.load()
            
            # Error should be related to YAML parsing
            assert "yaml" in str(exc_info.value).lower()
        finally:
            os.unlink(yaml_path)
    
    def test_yaml_provider_save_creates_directories(self):
        """Test that YamlFileProvider creates directories when saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = os.path.join(temp_dir, "nested", "config", "dir")
            config_path = os.path.join(config_dir, "config.yaml")
            
            # Directory should not exist yet
            assert not os.path.exists(config_dir)
            
            # Create provider and save config
            provider = YamlFileProvider(config_path)
            provider.save({"key": "value"})
            
            # Directory and file should now exist
            assert os.path.exists(config_dir)
            assert os.path.exists(config_path)
            
            # Load the saved config to verify
            assert provider.load() == {"key": "value"}


class TestEnvironmentProvider:
    """Tests for the EnvironmentProvider implementation."""
    
    def test_env_provider_contract(self):
        """Test that EnvironmentProvider implements the contract correctly."""
        provider = EnvironmentProvider(prefix="TEST")
        
        # Should implement load (save might raise NotImplementedError)
        assert hasattr(provider, "load")
        assert hasattr(provider, "save")
        
        # Prefix should be stored and accessible
        assert provider.prefix == "TEST"
    
    def test_env_provider_nested_keys(self):
        """Test that EnvironmentProvider handles nested keys correctly."""
        # Test with nested keys using __ separator
        with mock.patch.dict(os.environ, {
            "TEST_SIMPLE": "value",
            "TEST_NESTED__KEY": "nested_value",
            "TEST_DEEPLY__NESTED__KEY": "deeply_nested_value"
        }):
            provider = EnvironmentProvider(prefix="TEST")
            config = provider.load()
            
            # Config should have proper nesting
            assert config["simple"] == "value"
            assert config["nested"]["key"] == "nested_value"
            assert config["deeply"]["nested"]["key"] == "deeply_nested_value"
    
    def test_env_provider_key_transformation(self):
        """Test that EnvironmentProvider transforms keys correctly."""
        # Test with various key formats
        with mock.patch.dict(os.environ, {
            "TEST_LOWER_CASE": "lower",
            "TEST_CAMEL_CASE": "camel",
            "TEST_WITH_NUMBERS_123": "numbers"
        }):
            provider = EnvironmentProvider(prefix="TEST")
            config = provider.load()
            
            # Keys should be transformed to lowercase
            assert config["lower_case"] == "lower"
            assert config["camel_case"] == "camel"
            assert config["with_numbers_123"] == "numbers"
    
    def test_env_provider_prefix_handling(self):
        """Test that EnvironmentProvider handles prefixes correctly."""
        # Test with and without prefix
        with mock.patch.dict(os.environ, {
            "TEST_KEY": "test_value",
            "OTHER_KEY": "other_value"
        }):
            # With prefix
            test_provider = EnvironmentProvider(prefix="TEST")
            test_config = test_provider.load()
            
            # Only matching prefix should be included
            assert "key" in test_config
            assert test_config["key"] == "test_value"
            assert len(test_config) == 1
            
            # Different prefix
            other_provider = EnvironmentProvider(prefix="OTHER")
            other_config = other_provider.load()
            
            # Only matching prefix should be included
            assert "key" in other_config
            assert other_config["key"] == "other_value"
            assert len(other_config) == 1
            
            # No prefix
            all_provider = EnvironmentProvider(prefix="")
            all_config = all_provider.load()
            
            # All environment variables should be included (could be a lot)
            # Just check the ones we set
            assert "test_key" in all_config
            assert "other_key" in all_config