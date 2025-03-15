"""Tests for the ConfigManager interface contract.

This module focuses on testing that the ConfigManager interface adheres
to SOLID principles, particularly the Dependency Inversion Principle
and the Interface Segregation Principle.
"""

import os
import tempfile
from typing import Dict, Any, List, Protocol, Optional, Type, TypeVar, runtime_checkable
from unittest import mock

import pytest
from pydantic import BaseModel

from src.ember.core.configs.config_manager import ConfigManager, create_config_manager
from src.ember.core.configs.providers import ConfigProvider, YamlFileProvider, EnvironmentProvider
from src.ember.core.configs.transformer import ConfigTransformer
from src.ember.core.configs.exceptions import ConfigError
from src.ember.core.configs.schema import EmberConfig


# Minimal models for testing
class SimpleConfig(BaseModel):
    """Simple configuration model for testing."""
    name: str
    value: int = 0
    nested: Dict[str, Any] = {}


@runtime_checkable
class ConfigManagerProtocol(Protocol):
    """A protocol that the ConfigManager should satisfy."""
    
    def load(self) -> Any:
        """Load configuration."""
        ...
    
    def get_config(self) -> Any:
        """Get the current configuration."""
        ...
    
    def reload(self) -> Any:
        """Reload configuration."""
        ...
    
    def get(self, *path: str, default: Any = None) -> Any:
        """Get a configuration value by path."""
        ...


class TestConfigManagerContract:
    """Tests for the ConfigManager interface contract."""

    def test_config_manager_satisfies_protocol(self):
        """Test that ConfigManager satisfies the minimal protocol."""
        manager = ConfigManager(SimpleConfig)
        assert isinstance(manager, ConfigManagerProtocol)
    
    def test_dependency_inversion_principle(self):
        """Test that ConfigManager follows the Dependency Inversion Principle."""
        # Create mock provider and transformer
        mock_provider = mock.MagicMock(spec=ConfigProvider)
        mock_provider.load.return_value = {"name": "test", "value": 42}
        
        mock_transformer = mock.MagicMock(spec=ConfigTransformer)
        mock_transformer.transform.return_value = {"name": "transformed", "value": 99}
        
        # Create manager with the mock dependencies
        manager = ConfigManager(
            schema_class=SimpleConfig,
            providers=[mock_provider],
            transformers=[mock_transformer]
        )
        
        # Load configuration
        config = manager.load()
        
        # Verify that the manager used the dependencies correctly
        mock_provider.load.assert_called_once()
        mock_transformer.transform.assert_called_once()
        
        # Verify that the result is correctly validated
        assert isinstance(config, SimpleConfig)
        assert config.name == "transformed"
        assert config.value == 99
    
    def test_single_responsibility_principle(self):
        """Test that ConfigManager has a single, well-defined responsibility."""
        # ConfigManager's responsibility is to orchestrate configuration loading,
        # transformation, validation, and access.
        
        # Create a provider and transformer with specific responsibilities
        provider_data = {"name": "provider", "value": 1}
        provider = mock.MagicMock(spec=ConfigProvider)
        provider.load.return_value = provider_data
        
        def transform_func(config):
            config["value"] *= 2
            return config
            
        transformer = ConfigTransformer()
        transformer.add_transformation(transform_func)
        
        # Create manager
        manager = ConfigManager(
            schema_class=SimpleConfig,
            providers=[provider],
            transformers=[transformer]
        )
        
        # Load configuration
        config = manager.load()
        
        # Provider should handle loading
        provider.load.assert_called_once()
        assert provider.load.return_value is provider_data
        
        # ConfigManager should handle orchestration and validation
        assert isinstance(config, SimpleConfig)
        assert config.name == "provider"
        assert config.value == 2  # Transformed value
    
    def test_interface_segregation_principle(self):
        """Test that ConfigManager's interface is cohesive and minimal."""
        # Create a manager
        manager = ConfigManager(SimpleConfig)
        
        # Check that it has a cohesive set of methods
        assert hasattr(manager, "load")
        assert hasattr(manager, "get_config")
        assert hasattr(manager, "reload")
        assert hasattr(manager, "get")
        assert hasattr(manager, "save")
        
        # Check that it doesn't expose implementation details
        assert not hasattr(manager, "_providers")
        assert not hasattr(manager, "_transformers")
        assert not hasattr(manager, "_config")
    
    def test_open_closed_principle(self):
        """Test that ConfigManager follows the Open/Closed Principle."""
        # Create a basic manager
        manager = ConfigManager(SimpleConfig)
        
        # Create an extended manager without modifying the original class
        class EnhancedManager(ConfigManager):
            def __init__(self, schema_class, providers=None, transformers=None):
                super().__init__(schema_class, providers, transformers)
                self.enhancement_applied = True
            
            def get_enhanced_config(self):
                """Enhanced method that wasn't in the original."""
                config = self.get_config()
                return {"enhanced": True, "original": config.model_dump()}
        
        # The enhanced manager should work with the original interface
        enhanced = EnhancedManager(SimpleConfig)
        
        # And it should also have its enhancements
        assert hasattr(enhanced, "enhancement_applied")
        assert hasattr(enhanced, "get_enhanced_config")
    
    def test_liskov_substitution_principle(self):
        """Test that ConfigManager follows the Liskov Substitution Principle."""
        # Create a function that works with any ConfigManager
        def use_config_manager(manager: ConfigManager):
            # Should be able to call these methods on any ConfigManager
            config = manager.get_config()
            path_value = manager.get("nested", "key", default="default")
            return config, path_value
        
        # Test with a regular ConfigManager
        regular_manager = ConfigManager(
            SimpleConfig,
            providers=[mock.MagicMock(spec=ConfigProvider, load=mock.MagicMock(
                return_value={"name": "test", "value": 42, "nested": {"key": "value"}}
            ))]
        )
        
        # Test with a custom subclass
        class CustomManager(ConfigManager):
            def get_custom_value(self):
                return "custom"
        
        custom_manager = CustomManager(
            SimpleConfig,
            providers=[mock.MagicMock(spec=ConfigProvider, load=mock.MagicMock(
                return_value={"name": "custom", "value": 99, "nested": {"key": "custom_value"}}
            ))]
        )
        
        # Function should work with both managers
        regular_result, regular_path = use_config_manager(regular_manager)
        custom_result, custom_path = use_config_manager(custom_manager)
        
        # Results should be valid but different
        assert isinstance(regular_result, SimpleConfig)
        assert isinstance(custom_result, SimpleConfig)
        assert regular_result.name == "test"
        assert custom_result.name == "custom"
        assert regular_path == "value"
        assert custom_path == "custom_value"


class TestConfigManagerLifecycle:
    """Tests for the ConfigManager's lifecycle management."""
    
    def test_config_initialization(self):
        """Test that ConfigManager initializes properly."""
        # Should accept no providers/transformers
        manager = ConfigManager(SimpleConfig)
        assert manager._providers == []
        assert manager._transformers == []
        assert manager._config is None
        
        # Should accept custom providers/transformers
        provider = mock.MagicMock(spec=ConfigProvider)
        transformer = mock.MagicMock(spec=ConfigTransformer)
        
        manager = ConfigManager(
            SimpleConfig,
            providers=[provider],
            transformers=[transformer]
        )
        
        assert manager._providers == [provider]
        assert manager._transformers == [transformer]
    
    def test_config_loading(self):
        """Test the configuration loading process."""
        # Create a provider with test data
        provider = mock.MagicMock(spec=ConfigProvider)
        provider.load.return_value = {"name": "test", "value": 42}
        
        # Create manager
        manager = ConfigManager(SimpleConfig, providers=[provider])
        
        # Load configuration
        config = manager.load()
        
        # Verify loading process
        provider.load.assert_called_once()
        assert isinstance(config, SimpleConfig)
        assert config.name == "test"
        assert config.value == 42
        
        # Config should be cached
        assert manager._config is config
    
    def test_config_reloading(self):
        """Test configuration reloading process."""
        # Create a provider with changing data
        provider = mock.MagicMock(spec=ConfigProvider)
        provider.load.side_effect = [
            {"name": "first", "value": 1},
            {"name": "second", "value": 2}
        ]
        
        # Create manager
        manager = ConfigManager(SimpleConfig, providers=[provider])
        
        # First load
        config1 = manager.get_config()
        assert config1.name == "first"
        assert config1.value == 1
        
        # Reload
        config2 = manager.reload()
        
        # Should call provider.load again
        assert provider.load.call_count == 2
        
        # Should get new data
        assert config2.name == "second"
        assert config2.value == 2
        
        # Should be a different instance
        assert config1 is not config2
    
    def test_config_saving(self):
        """Test configuration saving process."""
        # Create a provider
        provider = mock.MagicMock(spec=ConfigProvider)
        provider.load.return_value = {"name": "original", "value": 1}
        
        # Create manager
        manager = ConfigManager(SimpleConfig, providers=[provider])
        
        # Load initial config
        config = manager.get_config()
        
        # Simulate modified config
        manager._config = SimpleConfig(name="modified", value=2)
        
        # Save the config
        manager.save()
        
        # Provider.save should be called with the updated config
        provider.save.assert_called_once()
        saved_config = provider.save.call_args[0][0]
        assert saved_config["name"] == "modified"
        assert saved_config["value"] == 2


class TestDefaultConfigManager:
    """Tests for the default config manager factory."""
    
    def test_create_config_manager(self):
        """Test the default config manager creation."""
        # Create a default manager
        with mock.patch.dict(os.environ, {"EMBER_CONFIG": "nonexistent.yaml"}):
            manager = create_config_manager()
        
        # Should be a ConfigManager instance
        assert isinstance(manager, ConfigManager)
        
        # Should have the correct schema
        assert manager._schema_class == EmberConfig
        
        # Should have YamlFileProvider and EnvironmentProvider
        assert len(manager._providers) == 2
        assert isinstance(manager._providers[0], YamlFileProvider)
        assert isinstance(manager._providers[1], EnvironmentProvider)
        
        # YAML provider should use the path from environment
        assert manager._providers[0].file_path == "nonexistent.yaml"
        
        # Environment provider should use the EMBER prefix
        assert manager._providers[1].prefix == "EMBER"
        
        # Should have at least one transformer
        assert len(manager._transformers) > 0
    
    def test_default_manager_with_custom_path(self):
        """Test default manager creation with custom path."""
        manager = create_config_manager(config_path="custom_path.yaml")
        
        # YAML provider should use the custom path
        assert manager._providers[0].file_path == "custom_path.yaml"


class TestConfigManagerThreadSafety:
    """Tests for the ConfigManager's thread safety guarantees."""
    
    def test_thread_safety_with_lock(self):
        """Test that ConfigManager properly uses locks."""
        # Create a manager
        manager = ConfigManager(SimpleConfig)
        
        # Mock the lock
        original_lock = manager._lock
        mock_lock = mock.MagicMock()
        mock_lock.__enter__ = mock.MagicMock()
        mock_lock.__exit__ = mock.MagicMock()
        manager._lock = mock_lock
        
        # Call methods that should use the lock
        try:
            manager.load()
            manager.get_config()
            manager.reload()
            manager.get("key")
            
            # Each method should acquire the lock
            assert mock_lock.__enter__.call_count == 4
            assert mock_lock.__exit__.call_count == 4
        finally:
            # Restore the original lock
            manager._lock = original_lock


class TestConfigManagerValidation:
    """Tests for the ConfigManager's validation functionality."""
    
    def test_validation_success(self):
        """Test successful validation."""
        provider = mock.MagicMock(spec=ConfigProvider)
        provider.load.return_value = {"name": "test", "value": 42}
        
        manager = ConfigManager(SimpleConfig, providers=[provider])
        config = manager.load()
        
        # Should be a valid SimpleConfig instance
        assert isinstance(config, SimpleConfig)
        assert config.name == "test"
        assert config.value == 42
    
    def test_validation_error(self):
        """Test validation error handling."""
        provider = mock.MagicMock(spec=ConfigProvider)
        provider.load.return_value = {"value": 42}  # Missing required 'name' field
        
        manager = ConfigManager(SimpleConfig, providers=[provider])
        
        # Should raise ConfigError
        with pytest.raises(ConfigError) as exc_info:
            manager.load()
        
        # Error should mention validation
        assert "validation" in str(exc_info.value).lower()
    
    def test_type_conversion(self):
        """Test automatic type conversion during validation."""
        provider = mock.MagicMock(spec=ConfigProvider)
        provider.load.return_value = {"name": "test", "value": "42"}  # String instead of int
        
        manager = ConfigManager(SimpleConfig, providers=[provider])
        config = manager.load()
        
        # String should be converted to int
        assert config.value == 42
        assert isinstance(config.value, int)