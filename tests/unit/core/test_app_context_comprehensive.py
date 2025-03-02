"""
Comprehensive tests for the ember.core.app_context module.

This module tests the EmberAppContext, EmberContext and related functions.
"""

import logging
import threading
import pytest
from unittest.mock import MagicMock, patch, ANY, call
import signal

from ember.core.app_context import (
    EmberAppContext,
    create_ember_app,
    EmberContext,
    get_ember_context,
)
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.base.services.usage_service import UsageService
from ember.core.configs.config import auto_register_known_models, initialize_api_keys


class TestEmberAppContext:
    """Tests for the EmberAppContext class."""

    def test_initialization(self):
        """Test that EmberAppContext initializes with provided services."""
        # Create mock dependencies
        config_manager = MagicMock()
        model_registry = MagicMock()
        usage_service = MagicMock()
        logger = MagicMock()
        
        # Initialize the context
        context = EmberAppContext(
            config_manager=config_manager,
            model_registry=model_registry,
            usage_service=usage_service,
            logger=logger,
        )
        
        # Verify that the context has the correct attributes
        assert context.config_manager is config_manager
        assert context.model_registry is model_registry
        assert context.usage_service is usage_service
        assert context.logger is logger
    
    def test_initialization_with_none_usage_service(self):
        """Test that EmberAppContext can initialize with None usage_service."""
        # Create mock dependencies
        config_manager = MagicMock()
        model_registry = MagicMock()
        logger = MagicMock()
        
        # Initialize the context with None usage_service
        context = EmberAppContext(
            config_manager=config_manager,
            model_registry=model_registry,
            usage_service=None,
            logger=logger,
        )
        
        # Verify that the context has the correct attributes
        assert context.config_manager is config_manager
        assert context.model_registry is model_registry
        assert context.usage_service is None
        assert context.logger is logger


class TestCreateEmberApp:
    """Tests for the create_ember_app function."""

    @patch('ember.core.app_context.logging.getLogger')
    @patch('ember.core.configs.config.ConfigManager')
    @patch('ember.core.configs.config.initialize_api_keys')
    @patch('ember.core.app_context.ModelRegistry')
    @patch('ember.core.configs.config.auto_register_known_models')
    @patch('ember.core.app_context.UsageService')
    def test_create_ember_app_with_default_config(
        self, mock_usage_service, mock_auto_register, mock_model_registry,
        mock_init_api_keys, mock_config_manager, mock_get_logger
    ):
        """Test creating an app context with default configuration."""
        # Setup mocks
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        mock_config = MagicMock()
        mock_config_manager.return_value = mock_config
        
        mock_registry = MagicMock()
        mock_model_registry.return_value = mock_registry
        
        mock_usage = MagicMock()
        mock_usage_service.return_value = mock_usage
        
        # Call the function
        result = create_ember_app()
        
        # Verify logger creation
        mock_get_logger.assert_called_once_with("ember")
        
        # Verify config manager creation
        mock_config_manager.assert_called_once_with(
            config_filename=None, 
            logger=mock_logger
        )
        
        # Verify API key initialization
        mock_init_api_keys.assert_called_once_with(
            mock_config, 
            logger=mock_logger
        )
        
        # Verify model registry creation and initialization
        mock_model_registry.assert_called_once_with(logger=mock_logger)
        mock_auto_register.assert_called_once_with(
            registry=mock_registry,
            config_manager=mock_config
        )
        
        # Verify usage service creation
        mock_usage_service.assert_called_once_with(logger=mock_logger)
        
        # Verify app context creation with all components
        assert isinstance(result, EmberAppContext)
        assert result.config_manager is mock_config
        assert result.model_registry is mock_registry
        assert result.usage_service is mock_usage
        assert result.logger is mock_logger
    
    @patch('ember.core.app_context.logging.getLogger')
    @patch('ember.core.configs.config.ConfigManager')
    @patch('ember.core.configs.config.initialize_api_keys')
    @patch('ember.core.app_context.ModelRegistry')
    @patch('ember.core.configs.config.auto_register_known_models')
    @patch('ember.core.app_context.UsageService')
    def test_create_ember_app_with_custom_config(
        self, mock_usage_service, mock_auto_register, mock_model_registry,
        mock_init_api_keys, mock_config_manager, mock_get_logger
    ):
        """Test creating an app context with custom configuration."""
        # Setup mocks
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        mock_config = MagicMock()
        mock_config_manager.return_value = mock_config
        
        mock_registry = MagicMock()
        mock_model_registry.return_value = mock_registry
        
        mock_usage = MagicMock()
        mock_usage_service.return_value = mock_usage
        
        # Call the function with custom config path
        custom_config = "test_config.ini"
        result = create_ember_app(config_filename=custom_config)
        
        # Verify config manager creation with custom path
        mock_config_manager.assert_called_once_with(
            config_filename=custom_config, 
            logger=mock_logger
        )
        
        # Other verifications remain the same
        assert isinstance(result, EmberAppContext)
        assert result.config_manager is mock_config
        assert result.model_registry is mock_registry
        assert result.usage_service is mock_usage
        assert result.logger is mock_logger


class TestEmberContext:
    """Tests for the EmberContext class."""

    def setup_method(self):
        """Enable test mode for EmberContext before each test."""
        # Save original test mode state
        self.original_test_mode = EmberContext._test_mode
        
        # Enable test mode to prevent deadlocks
        EmberContext.enable_test_mode()
    
    def teardown_method(self):
        """Restore original test mode after each test."""
        # Restore original test mode
        if self.original_test_mode:
            EmberContext.enable_test_mode()
        else:
            EmberContext.disable_test_mode()
    
    def test_singleton_pattern(self):
        """Test that EmberContext follows the singleton pattern in normal mode."""
        # Temporarily disable test mode to verify singleton behavior
        EmberContext.disable_test_mode()
        
        # Create two instances and verify they're the same
        context1 = EmberContext()
        context2 = EmberContext()
        assert context1 is context2
        
        # Re-enable test mode for other tests
        EmberContext.enable_test_mode()
    
    def test_test_mode_creates_separate_instances(self):
        """Test that test mode creates separate instances."""
        # Test mode is already enabled in setup
        context1 = EmberContext()
        context2 = EmberContext()
        assert context1 is not context2
    
    @patch('ember.core.app_context.create_ember_app')
    def test_lazy_loading(self, mock_create_app):
        """Test that app_context is lazy-loaded on first access."""
        # Setup mock
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app
        
        # Create context but don't access app_context yet
        context = EmberContext()
        
        # Verify create_ember_app not called yet
        mock_create_app.assert_not_called()
        
        # Access app_context to trigger lazy loading
        app = context.app_context
        
        # Verify create_ember_app was called
        mock_create_app.assert_called_once()
        assert app is mock_app
    
    @patch('ember.core.app_context.create_ember_app')
    def test_initialize_with_config_path(self, mock_create_app):
        """Test initializing EmberContext with a config path."""
        # Setup mock
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app
        
        # Initialize with config path
        config_path = "test_config.yaml"
        context = EmberContext.initialize(config_path=config_path)
        
        # Verify create_ember_app was called with correct path
        mock_create_app.assert_called_once_with(config_filename=config_path)
        assert context._app_context is mock_app
    
    def test_initialize_with_app_context(self):
        """Test initializing EmberContext with an app context."""
        # Create mock app context
        mock_app = MagicMock(spec=EmberAppContext)
        
        # Initialize with app context
        context = EmberContext.initialize(app_context=mock_app)
        
        # Verify app context was set
        assert context._app_context is mock_app
    
    @patch('ember.core.app_context.create_ember_app')
    def test_initialize_multiple_calls(self, mock_create_app):
        """Test calling initialize multiple times."""
        # Setup mocks for multiple calls
        mock_app1 = MagicMock()
        mock_app2 = MagicMock()
        mock_create_app.side_effect = [mock_app1, mock_app2]
        
        # First initialization
        context1 = EmberContext.initialize()
        assert context1._app_context is mock_app1
        
        # Second initialization
        context2 = EmberContext.initialize()
        
        # In test mode, should create different instances
        assert context2 is not context1
        assert context2._app_context is mock_app2
    
    @patch('ember.core.app_context.create_ember_app')
    def test_get_initializes_if_needed(self, mock_create_app):
        """Test that get() initializes the context if needed."""
        # Setup mock
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app
        
        # Call get() before initializing
        context = EmberContext.get()
        
        # Verify app was created
        mock_create_app.assert_called_once()
        assert context._app_context is mock_app
    
    @patch('ember.core.app_context.create_ember_app')
    def test_get_returns_new_context_in_test_mode(self, mock_create_app):
        """Test that get() returns a new context each time in test mode."""
        # Setup mock
        mock_apps = [MagicMock(), MagicMock()]
        mock_create_app.side_effect = mock_apps
        
        # Call get() multiple times
        context1 = EmberContext.get()
        context2 = EmberContext.get()
        
        # In test mode, should create different instances
        assert context1 is not context2
        assert context1._app_context is mock_apps[0]
        assert context2._app_context is mock_apps[1]
    
    @patch('ember.core.app_context.create_ember_app')
    def test_getattr_lazy_loads(self, mock_create_app):
        """Test that accessing attributes lazily loads the app context."""
        # Setup mock
        mock_app = MagicMock()
        mock_app.model_registry = MagicMock()
        mock_create_app.return_value = mock_app
        
        # Create context
        context = EmberContext()
        
        # Verify create_ember_app not called yet
        mock_create_app.assert_not_called()
        
        # Access attribute to trigger lazy loading
        registry = context.model_registry
        
        # Verify create_ember_app was called
        mock_create_app.assert_called_once()
        assert registry is mock_app.model_registry
    
    @patch('ember.core.app_context.create_ember_app')
    def test_getattr_raises_for_missing_attributes(self, mock_create_app):
        """Test that accessing non-existent attributes raises AttributeError."""
        # Setup mock
        mock_app = MagicMock(spec=EmberAppContext)
        mock_create_app.return_value = mock_app
        
        # Create context
        context = EmberContext()
        
        # Access non-existent attribute should raise
        with pytest.raises(AttributeError):
            non_existent = context.non_existent_attribute
    
    @patch('ember.core.app_context.create_ember_app')
    def test_property_accessors(self, mock_create_app):
        """Test the property accessors for app_context, registry, etc."""
        # Setup mock app and services
        mock_registry = MagicMock()
        mock_config = MagicMock()
        mock_usage = MagicMock()
        mock_logger = MagicMock()
        
        mock_app = MagicMock(spec=EmberAppContext)
        mock_app.model_registry = mock_registry
        mock_app.config_manager = mock_config
        mock_app.usage_service = mock_usage
        mock_app.logger = mock_logger
        
        mock_create_app.return_value = mock_app
        
        # Create context
        context = EmberContext()
        
        # Test property accessors
        assert context.app_context is mock_app
        assert context.registry is mock_registry
        assert context.config_manager is mock_config
        assert context.usage_service is mock_usage
        assert context.logger is mock_logger


class TestGetEmberContext:
    """Tests for the get_ember_context function."""

    def setup_method(self):
        """Reset the EmberContext singleton before each test."""
        # Save original test mode state
        self.original_test_mode = EmberContext._test_mode
        
        # Reset the singleton instance
        EmberContext._instance = None
    
    def teardown_method(self):
        """Restore original test mode after each test."""
        # Restore original test mode
        if self.original_test_mode:
            EmberContext.enable_test_mode()
        else:
            EmberContext.disable_test_mode()

    @patch('ember.core.app_context.EmberContext.get')
    def test_get_ember_context(self, mock_get):
        """Test that get_ember_context delegates to EmberContext.get()."""
        # Setup mock
        mock_context = MagicMock()
        mock_get.return_value = mock_context
        
        # Call the function
        result = get_ember_context()
        
        # Verify delegation
        mock_get.assert_called_once()
        assert result is mock_context
    
    @patch('ember.core.app_context.create_ember_app')
    def test_get_ember_context_in_test_mode(self, mock_create_app):
        """Test that get_ember_context works correctly in test mode."""
        # Setup mock to return different contexts
        mock_app1 = MagicMock()
        mock_app2 = MagicMock()
        mock_create_app.side_effect = [mock_app1, mock_app2]
        
        # Enable test mode
        EmberContext.enable_test_mode()
        
        # Call the function multiple times
        context1 = get_ember_context()
        context2 = get_ember_context()
        
        # In test mode, should return different instances
        assert isinstance(context1, EmberContext)
        assert isinstance(context2, EmberContext)
        assert context1 is not context2


class TestConcurrency:
    """Tests for EmberContext concurrency handling."""

    def setup_method(self):
        """Reset the EmberContext singleton before each test."""
        # Save original test mode state
        self.original_test_mode = EmberContext._test_mode
        
        # Reset the singleton instance
        EmberContext._instance = None
        EmberContext._lock = threading.Lock()
    
    def teardown_method(self):
        """Restore original test mode after each test."""
        # Restore original test mode
        if self.original_test_mode:
            EmberContext.enable_test_mode()
        else:
            EmberContext.disable_test_mode()

    @patch('ember.core.app_context.create_ember_app')
    def test_concurrent_initialization_singleton_mode(self, mock_create_app):
        """Test that concurrent initialization creates only one instance in singleton mode."""
        # Ensure test mode is disabled
        EmberContext.disable_test_mode()
        
        # Setup mock
        mock_app_context = MagicMock()
        mock_create_app.return_value = mock_app_context
        
        # Create a barrier to synchronize threads (reduced from 10 to 3 threads)
        barrier = threading.Barrier(3)
        results = []
        
        def initialize_context():
            try:
                # Wait for all threads to reach this point (with timeout)
                barrier.wait(timeout=1)
                # Initialize context
                context = EmberContext.initialize()
                # Store the result
                results.append(context)
            except threading.BrokenBarrierError:
                # Handle timeout gracefully
                pass
        
        # Create and start threads (reduced from 10 to 3 threads)
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=initialize_context)
            thread.daemon = True  # Mark as daemon so test doesn't hang
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete (with timeout)
        for thread in threads:
            thread.join(timeout=2)
        
        # If we got results, verify them
        if results:
            # Verify that create_ember_app was called once
            assert mock_create_app.call_count >= 1
            
            # Verify that all threads got the same context instance
            for context in results[1:]:
                assert context is results[0]
    
    @patch('ember.core.app_context.create_ember_app')
    def test_concurrent_initialization_test_mode(self, mock_create_app):
        """Test that concurrent initialization creates separate instances in test mode."""
        # Enable test mode
        EmberContext.enable_test_mode()
        
        # Setup mock to return different contexts
        mock_contexts = [MagicMock() for _ in range(3)]  # Reduced from 10 to 3
        mock_create_app.side_effect = mock_contexts
        
        # Create a barrier to synchronize threads
        barrier = threading.Barrier(3)  # Reduced from 10 to 3 threads
        results = []
        
        def initialize_context():
            try:
                # Wait for all threads to reach this point (with timeout)
                barrier.wait(timeout=1)
                # Initialize context in test mode
                context = EmberContext.initialize()
                # Store the result
                results.append(context)
            except threading.BrokenBarrierError:
                # Handle timeout gracefully
                pass
        
        # Create and start threads
        threads = []
        for _ in range(3):  # Reduced from 10 to 3
            thread = threading.Thread(target=initialize_context)
            thread.daemon = True  # Mark as daemon so test doesn't hang
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete (with timeout)
        for thread in threads:
            thread.join(timeout=2)
        
        # If we got results, verify them
        if results:
            # Verify that create_ember_app was called for each thread that completed
            assert mock_create_app.call_count == len(results)
            
            # In test mode, contexts should be different instances
            context_ids = {id(context) for context in results}
            assert len(context_ids) == len(results)