"""Configure pytest environment for all tests."""

import pytest
import sys
import os
from pathlib import Path
import importlib
import logging

logger = logging.getLogger(__name__)

# Get absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SRC_PATH = PROJECT_ROOT / "src"


# This is the critical path resolution that fixes the imports
# Create a modules dictionary to make ember.core accessible
class ImportFixerMeta(type):
    def __getattr__(cls, name):
        # Dynamically import modules from ember
        if name == "core":
            try:
                # Import src.ember.core
                module = importlib.import_module("src.ember.core")
                # Cache it for future access
                setattr(sys.modules["ember"], "core", module)
                # Also register it in sys.modules for direct imports
                sys.modules["ember.core"] = module
                return module
            except ImportError as e:
                print(f"Failed to import ember.core: {e}")
                raise
        raise AttributeError(f"Module 'ember' has no attribute '{name}'")


# Apply the metaclass to ember module
class EmberImportFixer(metaclass=ImportFixerMeta):
    pass


# Print current path for debugging
print(f"Unit test Python path: {sys.path}")
print(f"Unit test current directory: {os.getcwd()}")

# Add src directory first for proper imports
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(PROJECT_ROOT))

# Set up the import hook
sys.modules["ember"] = EmberImportFixer

# Create explicit mapping for core modules
CORE_MODULES = [
    "ember.core",
    "ember.core.non",
    "ember.core.registry",
    "ember.core.utils",
    "ember.core.registry.model",
    "ember.core.registry.operator",
    "ember.core.registry.prompt_specification",
    "ember.xcs",
]

# Pre-load critical modules
for module_name in CORE_MODULES:
    src_module_name = module_name.replace("ember.", "src.ember.")
    try:
        module = importlib.import_module(src_module_name)
        sys.modules[module_name] = module
    except ImportError:
        pass


@pytest.fixture(scope="session", autouse=True)
def global_setup_teardown():
    """
    Global fixture for session-level setup/teardown.
    - Can configure logging or environment variables here.
    """
    # TODO: placeholder to add global config as needed
    yield
    # TODO: placeholder to add global teardown as needed


@pytest.fixture
def mock_lm_generation(mocker):
    """
    Mocks LMModule model_instance.generate calls to return predictable responses.
    Ensures deterministic tests regardless of input prompt.
    """

    def mock_generate(prompt, temperature=1.0, max_tokens=None):
        return f"Mocked response: {prompt}, temp={temperature}"

    # Patch the DummyModel generate method.
    # Adjust the patch path if needed to reflect actual code location of get_model_registry usage.
    mocker.patch(
        "tests.get_model_registry().DummyModel.generate", side_effect=mock_generate
    )
