#!/usr/bin/env python
"""
Fuzzing test for ember.core.app_context module.

This fuzzes the EmberAppContext and EmberContext classes to identify any
edge cases or threading issues.
"""

import os
import sys
import time
import atheris
import threading
import tempfile
from typing import Optional, List, Dict, Any
from unittest.mock import MagicMock, patch

# Ensure Ember is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.ember.core.app_context import (
    EmberAppContext,
    EmberContext,
    get_ember_context,
)


class MockRegistry:
    """Mock implementation of ModelRegistry for fuzzing."""
    
    def __init__(self):
        """Initialize the mock registry."""
        self.models = {}
    
    def register_model(self, model_id, model_config):
        """Register a model in the registry."""
        self.models[model_id] = model_config
    
    def get_model(self, model_id):
        """Get a model from the registry."""
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not found")
        return self.models[model_id]


class MockConfigManager:
    """Mock implementation of ConfigManager for fuzzing."""
    
    def __init__(self):
        """Initialize the mock config manager."""
        self.config = {}
    
    def get_config(self, section, key, default=None):
        """Get a config value."""
        if section not in self.config:
            return default
        if key not in self.config[section]:
            return default
        return self.config[section][key]
    
    def set_config(self, section, key, value):
        """Set a config value."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value


class MockUsageService:
    """Mock implementation of UsageService for fuzzing."""
    
    def __init__(self):
        """Initialize the mock usage service."""
        self.usage = {}
    
    def track_usage(self, model_id, tokens_in, tokens_out):
        """Track model usage."""
        if model_id not in self.usage:
            self.usage[model_id] = {"tokens_in": 0, "tokens_out": 0}
        self.usage[model_id]["tokens_in"] += tokens_in
        self.usage[model_id]["tokens_out"] += tokens_out


class MockLogger:
    """Mock implementation of Logger for fuzzing."""
    
    def __init__(self):
        """Initialize the mock logger."""
        self.logs = []
    
    def debug(self, message):
        """Log a debug message."""
        self.logs.append(("DEBUG", message))
    
    def info(self, message):
        """Log an info message."""
        self.logs.append(("INFO", message))
    
    def warning(self, message):
        """Log a warning message."""
        self.logs.append(("WARNING", message))
    
    def error(self, message):
        """Log an error message."""
        self.logs.append(("ERROR", message))


def fuzz_app_context(data):
    """Fuzz the app context module."""
    fdp = atheris.FuzzedDataProvider(data)
    
    # Reset the EmberContext singleton for each fuzzing iteration
    EmberContext._instance = None
    EmberContext._lock = threading.Lock()
    
    # Create mock components
    config_manager = MockConfigManager()
    model_registry = MockRegistry()
    usage_service = MockUsageService()
    logger = MockLogger()
    
    # Fuzz the EmberAppContext
    app_context = EmberAppContext(
        config_manager=config_manager,
        model_registry=model_registry,
        usage_service=usage_service,
        logger=logger,
    )
    
    # Test initialization of EmberContext
    # Choose between different initialization methods
    init_method = fdp.ConsumeIntInRange(0, 2)
    
    if init_method == 0:
        # Initialize with default params
        context = EmberContext()
        # Force app_context initialization
        _ = context.app_context
    elif init_method == 1:
        # Initialize with config path
        context = EmberContext.initialize(config_path=fdp.ConsumeString(20))
    else:
        # Initialize with app_context
        context = EmberContext.initialize(app_context=app_context)
    
    # Access properties and services
    _ = context.registry
    _ = context.config_manager
    _ = context.usage_service
    _ = context.logger
    
    # Test attribute access
    try:
        attr_name = fdp.ConsumeString(20)
        if hasattr(app_context, attr_name):
            _ = getattr(context, attr_name)
    except AttributeError:
        # Expected for invalid attributes
        pass
    
    # Test concurrent access
    num_threads = fdp.ConsumeIntInRange(1, 5)
    barrier = threading.Barrier(num_threads + 1)  # +1 for main thread
    
    def access_context():
        # Wait for all threads to reach this point
        barrier.wait()
        # Get the context
        ctx = get_ember_context()
        # Access something to trigger lazy loading
        _ = ctx.registry
    
    # Create and start threads
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=access_context)
        thread.start()
        threads.append(thread)
    
    # Join main thread to barrier
    barrier.wait()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()


def run_fuzzer(time_limit: Optional[int] = None):
    """Run the fuzzer with the specified time limit or default iterations."""
    atheris.instrument_all()
    
    if time_limit:
        atheris.Setup(sys.argv, fuzz_app_context, time_limit=time_limit)
    else:
        atheris.Setup(sys.argv, fuzz_app_context)
        
    atheris.Fuzz()


if __name__ == "__main__":
    run_fuzzer()