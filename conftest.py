"""
Root conftest.py for pytest configuration
"""

import pytest
import sys
import os
from pathlib import Path

# Get absolute paths
PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_PATH = PROJECT_ROOT / "src"

print(f"Setting up Python paths in root conftest.py")

# Add src directory to path
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(PROJECT_ROOT))

# Configure asyncio as early as possible
pytest_plugins = ["pytest_asyncio"]

# Modify sys.modules to make src.ember packages available as ember packages
import sys
import types
import importlib.util


def map_module(from_name, to_name):
    """Map a module from one name to another in sys.modules."""
    if from_name in sys.modules:
        sys.modules[to_name] = sys.modules[from_name]


# Set up module aliases
ember_modules = [
    "src.ember",
    "src.ember.core",
    "src.ember.core.registry",
    "src.ember.core.registry.model",
    "src.ember.core.registry.operator",
    "src.ember.core.registry.prompt_specification",
    "src.ember.xcs",
    "src.ember.plugin_system",  # Add plugin_system module
]

# Create ember package and ember.core if they don't exist
if "ember" not in sys.modules:
    sys.modules["ember"] = types.ModuleType("ember")
if "ember.core" not in sys.modules:
    sys.modules["ember.core"] = types.ModuleType("ember.core")

# Map all relevant modules
for module_name in ember_modules:
    target_name = module_name.replace("src.", "")
    try:
        # Try to import the source module
        if module_name not in sys.modules:
            spec = importlib.util.find_spec(module_name)
            if spec:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

        # Map it to the target name
        if module_name in sys.modules:
            sys.modules[target_name] = sys.modules[module_name]
            print(f"Mapped {module_name} -> {target_name}")
    except ImportError:
        print(f"Could not import {module_name}")

# Special handling for plugin_system.py
try:
    # Direct import of plugin_system.py
    plugin_system_path = str(PROJECT_ROOT / "src" / "ember" / "plugin_system.py")
    spec = importlib.util.spec_from_file_location(
        "ember.plugin_system", plugin_system_path
    )
    if spec:
        plugin_system_module = importlib.util.module_from_spec(spec)
        sys.modules["ember.plugin_system"] = plugin_system_module
        spec.loader.exec_module(plugin_system_module)
        print("Explicitly loaded ember.plugin_system")
except Exception as e:
    print(f"Error loading plugin_system.py: {e}")


# No need to define custom command line options here as they're defined in tests/unit/xcs/transforms/conftest.py

# Configure pytest-asyncio to use session-scoped event loops by default
def pytest_configure(config):
    """Configure pytest-asyncio."""
    import pytest_asyncio

    pytest_asyncio.LOOP_SCOPE = "session"


# Add pytest.config helper to access the config during tests
@pytest.fixture(scope="session", autouse=True)
def _add_config_helper(request):
    """Add config attribute to pytest module for backward compatibility."""
    pytest.config = request.config


# Custom event loop policy fixture - replaces the deprecated event_loop fixture
@pytest.fixture(scope="session")
def event_loop_policy():
    """Return the event loop policy to use."""
    import asyncio

    return asyncio.get_event_loop_policy()
