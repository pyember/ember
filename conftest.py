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
import os
from pathlib import Path


def map_module(from_name, to_name):
    """Map a module from one name to another in sys.modules."""
    if from_name in sys.modules:
        sys.modules[to_name] = sys.modules[from_name]


# Set up module aliases - including explicitly listing operator.base and related modules
ember_modules = [
    "src.ember",
    "src.ember.core",
    "src.ember.core.registry",
    "src.ember.core.registry.model",
    "src.ember.core.registry.operator",
    "src.ember.core.registry.operator.base",  # Explicitly list operator.base
    "src.ember.core.registry.operator.base.operator_base",  # Explicitly list operator_base module
    "src.ember.core.registry.operator.core",
    "src.ember.core.registry.operator.core.ensemble",
    "src.ember.core.registry.operator.core.most_common",
    "src.ember.core.registry.operator.core.synthesis_judge",
    "src.ember.core.registry.operator.core.verifier",
    "src.ember.core.registry.specification",
    "src.ember.core.registry.model.base",
    "src.ember.core.registry.model.base.registry",
    "src.ember.core.registry.model.base.schemas",
    "src.ember.core.registry.model.base.services",
    "src.ember.core.registry.model.config",
    "src.ember.core.registry.model.providers",
    "src.ember.xcs",
    "src.ember.xcs.api",
    "src.ember.xcs.transforms",
    "src.ember.xcs.transforms.mesh",
    "src.ember.plugin_system",
    "src.ember.operator",
    "src.ember.non",
    "src.ember.models",
    "src.ember.data",
]

# Create all ember modules
for module_path in ember_modules:
    target_name = module_path.replace("src.", "")
    parts = target_name.split(".")
    current = ""
    
    # Create parent modules first
    for part in parts:
        if not current:
            current = part
        else:
            current = f"{current}.{part}"
        
        if current not in sys.modules:
            sys.modules[current] = types.ModuleType(current)

# More reliable direct mapping for critical modules
SRC_DIR = Path(PROJECT_ROOT) / "src"

# Map all relevant modules with extra error handling
for module_name in ember_modules:
    target_name = module_name.replace("src.", "")
    try:
        # Try to import the source module
        if module_name not in sys.modules:
            spec = importlib.util.find_spec(module_name)
            if spec:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                except ImportError as ie:
                    print(f"Warning: Could not fully load {module_name}: {ie}")
                    # Still keep the partial module

        # Map it to the target name
        if module_name in sys.modules:
            sys.modules[target_name] = sys.modules[module_name]
            print(f"Mapped {module_name} -> {target_name}")
    except ImportError:
        print(f"Could not import {module_name}")

# Special handling for critical modules that need direct loading
critical_modules = [
    ("ember.core.registry.operator.base.operator_base", SRC_DIR / "ember/core/registry/operator/base/operator_base.py"),
    ("ember.core.registry.operator.base._module", SRC_DIR / "ember/core/registry/operator/base/_module.py"),
]

for module_name, file_path in critical_modules:
    if os.path.exists(file_path):
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                print(f"Directly loaded {module_name} from {file_path}")
        except Exception as e:
            print(f"Error loading {module_name}: {e}")

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

# Special handling for xcs.py
try:
    # Direct import of xcs.py
    xcs_path = str(PROJECT_ROOT / "src" / "ember" / "xcs.py")
    spec = importlib.util.spec_from_file_location("ember.xcs", xcs_path)
    if spec:
        xcs_module = importlib.util.module_from_spec(spec)
        sys.modules["ember.xcs"] = xcs_module
        spec.loader.exec_module(xcs_module)
        print("Explicitly loaded ember.xcs")
except Exception as e:
    print(f"Error loading xcs.py: {e}")

# Special handling for non.py
try:
    # Direct import of non.py
    non_path = str(PROJECT_ROOT / "src" / "ember" / "non.py")
    spec = importlib.util.spec_from_file_location("ember.non", non_path)
    if spec:
        non_module = importlib.util.module_from_spec(spec)
        sys.modules["ember.non"] = non_module
        spec.loader.exec_module(non_module)
        print("Explicitly loaded ember.non")
except Exception as e:
    print(f"Error loading non.py: {e}")

# Special handling for operator module
try:
    # Load the operator module
    operator_init_path = str(PROJECT_ROOT / "src" / "ember" / "core" / "registry" / "operator" / "__init__.py")
    spec = importlib.util.spec_from_file_location("ember.core.registry.operator", operator_init_path)
    if spec:
        operator_module = importlib.util.module_from_spec(spec)
        sys.modules["ember.core.registry.operator"] = operator_module
        spec.loader.exec_module(operator_module)
        print("Explicitly loaded ember.core.registry.operator")
except Exception as e:
    print(f"Error loading operator module: {e}")

# Setup proper module access paths for imports
try:
    # Make sure the basic module namespace exists
    if "ember" not in sys.modules:
        sys.modules["ember"] = types.ModuleType("ember")
        
    # Define the basic module tree structure
    module_paths = [
        # Core modules
        "ember.core",
        "ember.core.registry",
        # Operator modules
        "ember.core.registry.operator",
        "ember.core.registry.operator.base",
        "ember.core.registry.operator.core",
        # Model modules
        "ember.core.registry.model",
        "ember.core.registry.model.base",
        "ember.core.registry.model.base.registry",
        "ember.core.registry.model.base.schemas",
        "ember.core.registry.model.base.services",
        # XCS modules
        "ember.xcs",
        "ember.xcs.api",
        "ember.xcs.transforms",
        # Simplified modules
        "ember.operator",
        "ember.non",
        "ember.models",
        "ember.data"
    ]
    
    # Ensure all basic modules exist in sys.modules
    for module_path in module_paths:
        if module_path not in sys.modules:
            sys.modules[module_path] = types.ModuleType(module_path)
            
    # Make the simplified imports modules point to their core counterparts
    # We only setup the basics here; tests should import directly from core paths
    
    # This approach ensures proper module resolution without extensive stubs
    
except Exception as e:
    print(f"Warning: Error setting up module paths: {e}")


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
