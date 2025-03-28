"""
Root conftest.py for pytest configuration
"""

import importlib
import logging
import os
import sys
from pathlib import Path

import pytest


# Patch Python's logging handlers to gracefully handle closed streams at process shutdown
# This prevents "I/O operation on closed file" errors from HTTP libraries during pytest teardown
def _patch_logging_for_shutdown():
    """
    Configure logging to handle closed streams during Python interpreter shutdown.

    This patch prevents "I/O operation on closed file" errors that occur when HTTP clients
    are garbage collected during interpreter shutdown after logging handlers have been closed.
    """
    if not hasattr(logging.StreamHandler, "_ember_patched"):
        # Save the original emit method
        original_emit = logging.StreamHandler.emit

        # Create a version that swallows "closed file" errors during shutdown
        def safe_emit(self, record):
            """StreamHandler.emit that handles closed streams during shutdown."""
            try:
                original_emit(self, record)
            except ValueError as e:
                if "closed file" not in str(e).lower():
                    raise
            except (IOError, OSError) as e:
                if "closed" not in str(e).lower():
                    raise

        # Apply the patch
        logging.StreamHandler.emit = safe_emit
        logging.StreamHandler._ember_patched = True

        # Also configure the specific loggers from httpcore that cause the issues
        for name in ["httpcore.connection", "httpcore.http11"]:
            logging.getLogger(name).setLevel(logging.INFO)


# Apply the patch as early as possible
_patch_logging_for_shutdown()

# Get absolute paths
PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_PATH = PROJECT_ROOT / "src"

print(f"Unit test Python path: {sys.path}")
print(f"Unit test current directory: {os.getcwd()}")

# Add src directory to path
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(PROJECT_ROOT))

# Configure asyncio as early as possible
pytest_plugins = ["pytest_asyncio"]

# Setup the Python import system for proper namespace imports
#
# This is the principled approach to handling imports for testing.
# We create proper package structure that matches production usage while
# allowing the code to be run from the source tree.


def setup_module_hierarchy():
    """Set up the proper module hierarchy for testing.

    This function ensures that modules like 'ember', 'ember.core', etc. are properly
    available in sys.modules, preserving proper importing semantics.

    It creates a proper module hierarchy that:
    1. Creates empty parent modules first
    2. Populates them with submodules in dependency order
    3. Manages import dependencies properly
    """
    import types

    # Register ember as a package
    if "ember" not in sys.modules:
        ember_pkg = types.ModuleType("ember")
        ember_pkg.__path__ = [str(SRC_PATH / "ember")]
        ember_pkg.__package__ = "ember"
        ember_pkg.__file__ = str(SRC_PATH / "ember" / "__init__.py")
        sys.modules["ember"] = ember_pkg
        print("Created ember package in sys.modules")

    # Define and register core subpackages (ensures they exist as packages)
    subpackages = [
        "core",
        "xcs",
        "api",
        "core.registry",
        "core.utils",
        "core.types",
        "core.config",
        "xcs.tracer",
        "core.registry.model",
        "core.registry.operator",
        "core.utils.data",
        "core.utils.eval",
    ]

    for subpkg in subpackages:
        full_pkg_name = f"ember.{subpkg}"
        if full_pkg_name not in sys.modules:
            # Create package module
            pkg = types.ModuleType(full_pkg_name)
            pkg.__package__ = full_pkg_name
            pkg.__path__ = [str(SRC_PATH / "ember" / subpkg.replace(".", "/"))]
            pkg.__file__ = str(
                SRC_PATH / "ember" / subpkg.replace(".", "/") / "__init__.py"
            )
            sys.modules[full_pkg_name] = pkg
            print(f"Created {full_pkg_name} package in sys.modules")

    # Now load the actual module implementations in proper dependency order
    modules_to_load = [
        # Load in a dependency-aware order
        "ember.core",
        "ember.xcs",
        "ember.api",
        "ember",  # Load the top-level last as it depends on submodules
    ]

    for module_name in modules_to_load:
        src_path = "src." + module_name
        try:
            # Import real implementation
            implementation = importlib.import_module(src_path)

            # Update or replace the placeholder package
            target_module = sys.modules[module_name]
            target_module.__dict__.update(implementation.__dict__)

            # Copy special attributes
            for attr in [
                "__all__",
                "__version__",
                "__path__",
                "__file__",
                "__package__",
            ]:
                if hasattr(implementation, attr):
                    setattr(target_module, attr, getattr(implementation, attr))

            print(f"Loaded implementation for {module_name}")
        except ImportError as e:
            print(f"Warning: Could not load implementation for {module_name}: {e}")


# No module hierarchy setup function defined yet


# Setup module alias for API components
# The API module is imported carefully due to its dependencies
# We need to set up the module hierarchy correctly to avoid import issues
def setup_api_modules():
    """Set up the API module hierarchy with proper dependencies.

    This function ensures that all API modules are imported in the correct order,
    with proper dependency management to avoid circular imports.
    """
    # First, import and set up core dependencies that the API module needs
    core_deps = [
        ("src.ember.core.utils.data.base.models", "ember.core.utils.data.base.models"),
        ("src.ember.core.utils.data.registry", "ember.core.utils.data.registry"),
        ("src.ember.core.utils.data.service", "ember.core.utils.data.service"),
        ("src.ember.core.utils.eval", "ember.core.utils.eval"),
        (
            "src.ember.core.registry.model.base.schemas",
            "ember.core.registry.model.base.schemas",
        ),
    ]

    for src_path, target_path in core_deps:
        try:
            module = importlib.import_module(src_path)
            sys.modules[target_path] = module
            print(f"Mapped dependency {src_path} -> {target_path}")
        except ImportError as e:
            print(f"Warning: Could not import API dependency {src_path}: {e}")

    # Now import the main API module
    module_path = "src.ember.api"
    target_path = "ember.api"
    api_module_loaded = False

    try:
        module = importlib.import_module(module_path)
        sys.modules[target_path] = module
        api_module_loaded = True
        print(f"Mapped {module_path} -> {target_path}")

        # Import individual submodules in a specific order to handle dependencies
        # Order matters here - modules with fewer dependencies first
        api_submodules = [
            "types",  # Basic types, no dependencies
            "xcs",  # XCS functionality
            "models",  # Model registry access
            "non",  # Network of Networks
            "operator",  # Basic operator interface
            "operators",  # Extended operator functionality
            "data",  # Data functionality (depends on core utils)
            "eval",  # Evaluation (depends on most other modules)
        ]

        for submodule in api_submodules:
            try:
                sub_path = f"{module_path}.{submodule}"
                sub_target = f"{target_path}.{submodule}"
                sub_module = importlib.import_module(sub_path)
                sys.modules[sub_target] = sub_module

                # Also import any exposed classes/functions
                # This makes from X import Y style imports work properly
                if hasattr(module, submodule):
                    # Get the submodule from the main module
                    parent_submodule = getattr(module, submodule)

                    # Make it available in the sys.modules map
                    sys.modules[sub_target] = parent_submodule

                print(f"  Mapped {sub_path} -> {sub_target}")
            except ImportError as e:
                print(f"  Warning: Could not import API submodule {sub_path}: {e}")

        return api_module_loaded
    except ImportError as e:
        print(f"Warning: Could not import API module {module_path}: {e}")
        return False


# Run the API module setup
api_module_loaded = setup_api_modules()

# Setup aliases for specific failing modules
try:
    # Setup app_context module
    from src.ember.core import app_context

    sys.modules["ember.core.app_context"] = app_context

    # Core model registry modules
    registry_model = importlib.import_module("src.ember.core.registry.model")
    sys.modules["ember.core.registry.model"] = registry_model

    # Import the base registry module to fix test issues
    from src.ember.core.registry.model.base.registry import (
        discovery,
        factory,
        model_registry,
    )

    sys.modules[
        "ember.core.registry.model.base.registry.model_registry"
    ] = model_registry
    sys.modules["ember.core.registry.model.base.registry.discovery"] = discovery
    sys.modules["ember.core.registry.model.base.registry.factory"] = factory

    # Import and map the plugin system
    from src.ember import plugin_system

    sys.modules["ember.plugin_system"] = plugin_system

    # Set up test provider classes for testing
    from typing import Any

    from src.ember.core.registry.model.base.schemas.chat_schemas import (
        ChatRequest,
        ChatResponse,
    )
    from src.ember.core.registry.model.providers.base_provider import BaseProviderModel

    # Create proper test providers
    class DummyServiceProvider(BaseProviderModel):
        """Test provider class for unit tests."""

        PROVIDER_NAME = "DummyService"

        def create_client(self) -> Any:
            """Return a simple mock client."""
            return self

        def forward(self, request: ChatRequest) -> ChatResponse:
            """Process the request and return a response."""
            return ChatResponse(data=f"Echo: {request.prompt}")

    class DummyAsyncProvider(BaseProviderModel):
        """Test async provider class for unit tests."""

        PROVIDER_NAME = "DummyAsyncService"

        def create_client(self) -> Any:
            """Return a simple mock client."""
            return self

        def forward(self, request: ChatRequest) -> ChatResponse:
            """Process the request and return a response."""
            return ChatResponse(data=f"Async Echo: {request.prompt}")

        async def __call__(self, prompt: str, **kwargs: Any) -> ChatResponse:
            """Override to make this an async callable."""
            chat_request: ChatRequest = ChatRequest(prompt=prompt, **kwargs)
            return self.forward(request=chat_request)

    class DummyErrorProvider(BaseProviderModel):
        """Test provider that deliberately raises errors."""

        PROVIDER_NAME = "DummyErrorService"

        def create_client(self) -> Any:
            """Return a simple mock client."""
            return self

        def forward(self, request: ChatRequest) -> ChatResponse:
            """Always raise an error when called."""
            raise RuntimeError("Dummy error triggered")

    # Register test providers in the global provider registry
    from src.ember import plugin_system

    plugin_system.registered_providers["DummyService"] = DummyServiceProvider
    plugin_system.registered_providers["DummyAsyncService"] = DummyAsyncProvider
    plugin_system.registered_providers["DummyErrorService"] = DummyErrorProvider

    # Import the schemas module to fix test issues
    from src.ember.core.registry.model.base.schemas import chat_schemas

    sys.modules["ember.core.registry.model.base.schemas.chat_schemas"] = chat_schemas

    # Import the services module to fix test issues
    from src.ember.core.registry.model.base.services import model_service, usage_service

    sys.modules["ember.core.registry.model.base.services.usage_service"] = usage_service
    sys.modules["ember.core.registry.model.base.services.model_service"] = model_service

    # Import utilities module
    from src.ember.core.utils.data import (
        initialization,
        loader_factory,
        # metadata_registry is deprecated, use registry instead
        registry,
    )

    # Create a stub for backward compatibility
    class MetadataRegistryStub:
        """Stub for the deprecated metadata_registry module."""

        # Add any symbols needed by tests
        DatasetRegistry = registry.DatasetRegistry
        DATASET_REGISTRY = registry.DATASET_REGISTRY

    # Set up module references
    sys.modules["ember.core.utils.data.registry"] = registry
    sys.modules["ember.core.utils.data.metadata_registry"] = MetadataRegistryStub()
    sys.modules["ember.core.utils.data.initialization"] = initialization
    sys.modules["ember.core.utils.data.loader_factory"] = loader_factory

    # Import additional utility modules that are needed for tests
    try:
        from src.ember.core.utils import embedding_utils, eval, retry_utils

        sys.modules["ember.core.utils.retry_utils"] = retry_utils
        sys.modules["ember.core.utils.embedding_utils"] = embedding_utils
        sys.modules["ember.core.utils.eval"] = eval

        # Import eval submodules
        from src.ember.core.utils.eval import pipeline

        sys.modules["ember.core.utils.eval.pipeline"] = pipeline
    except ImportError as e:
        print(f"Warning: Could not import utility modules: {e}")

    # Import the base models module
    from src.ember.core.utils.data.base import loaders, models, preppers

    sys.modules["ember.core.utils.data.base.models"] = models
    sys.modules["ember.core.utils.data.base.loaders"] = loaders
    sys.modules["ember.core.utils.data.base.preppers"] = preppers

    # Import datasets registry
    from src.ember.core.utils.data.datasets_registry import (
        aime,
        codeforces,
        commonsense_qa,
        gpqa,
        halueval,
        mmlu,
        short_answer,
        truthful_qa,
    )

    sys.modules["ember.core.utils.data.datasets_registry.aime"] = aime
    sys.modules["ember.core.utils.data.datasets_registry.codeforces"] = codeforces
    sys.modules["ember.core.utils.data.datasets_registry.gpqa"] = gpqa
    sys.modules["ember.core.utils.data.datasets_registry.truthful_qa"] = truthful_qa
    sys.modules["ember.core.utils.data.datasets_registry.mmlu"] = mmlu
    sys.modules[
        "ember.core.utils.data.datasets_registry.commonsense_qa"
    ] = commonsense_qa
    sys.modules["ember.core.utils.data.datasets_registry.halueval"] = halueval
    sys.modules["ember.core.utils.data.datasets_registry.short_answer"] = short_answer

    print("Successfully mapped additional modules for tests")
except Exception as e:
    print(f"Error mapping additional modules: {e}")

# Suppress irrelevant warnings during tests
import warnings

warnings.filterwarnings("ignore", message=".*XCS functionality partially unavailable.*")


# Configure pytest-asyncio to use session-scoped event loops by default
def pytest_configure(config):
    """Configure pytest-asyncio and register custom marks."""
    import pytest_asyncio

    pytest_asyncio.LOOP_SCOPE = "session"

    # Register custom marks used in tests
    config.addinivalue_line(
        "markers", "discovery: mark tests that interact with model discovery"
    )
    config.addinivalue_line("markers", "xcs: mark tests related to XCS functionality")
    config.addinivalue_line(
        "markers", "performance: mark tests that measure performance characteristics"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Modify test items based on command line options."""
    run_all = config.getoption("--run-all-tests")
    run_api = config.getoption("--run-api-tests")
    run_perf = config.getoption("--run-perf-tests")

    for item in items:
        skip_marks = [mark for mark in item.own_markers if mark.name == "skip"]
        skipif_marks = [mark for mark in item.own_markers if mark.name == "skipif"]

        # Special handling for different types of tests
        if any(mark.name == "performance" for mark in item.own_markers):
            # Only remove skip marks for performance tests if --run-perf-tests or --run-all-tests is specified
            if run_all or run_perf:
                for mark in skip_marks:
                    item.own_markers.remove(mark)
        elif any("API_KEY" in str(mark.args) for mark in skipif_marks):
            # Only remove skip marks for API tests if --run-api-tests is specified
            if run_api:
                for mark in skip_marks:
                    item.own_markers.remove(mark)
        elif run_all:
            # Remove skip marks for all other tests if --run-all-tests is specified
            for mark in skip_marks:
                item.own_markers.remove(mark)


# Setup specific API symbols that are imported directly by tests
def setup_api_symbols():
    """Setup common symbols from API modules that are directly imported by tests.

    This makes direct imports like `from ember.api import DatasetBuilder` work correctly.
    """
    if not api_module_loaded:
        print("Warning: API module not loaded, cannot set up API symbols")
        return

    # Import API module
    import src.ember.api as api_module

    # Get all the symbols defined in the API's __all__
    all_symbols = getattr(api_module, "__all__", [])

    # Map symbols from the API module to the ember.api namespace
    for symbol_name in all_symbols:
        if hasattr(api_module, symbol_name):
            symbol = getattr(api_module, symbol_name)
            # Add a reference directly to the ember.api module in sys.modules
            setattr(sys.modules["ember.api"], symbol_name, symbol)
            print(f"  Registered API symbol: {symbol_name}")

    # Some symbols require special handling due to their import locations
    # These are the problematic ones in the tests
    special_symbols = {
        # Classes from data.py
        "DatasetBuilder": ("src.ember.api.data", "DatasetBuilder"),
        "Dataset": ("src.ember.api.data", "Dataset"),
        "DatasetEntry": ("src.ember.core.utils.data.base.models", "DatasetEntry"),
        "TaskType": ("src.ember.core.utils.data.base.models", "TaskType"),
        # Functions/classes from eval.py
        "Evaluator": ("src.ember.api.eval", "Evaluator"),
        "EvaluationPipeline": ("src.ember.api.eval", "EvaluationPipeline"),
    }

    # Import and register each special symbol
    for symbol_name, (module_path, class_name) in special_symbols.items():
        try:
            # Import the module
            module = importlib.import_module(module_path)

            # Get the symbol from the module
            if hasattr(module, class_name):
                symbol = getattr(module, class_name)

                # Register in the API module
                setattr(sys.modules["ember.api"], symbol_name, symbol)
                print(f"  Registered special API symbol: {symbol_name}")
        except (ImportError, AttributeError) as e:
            print(f"  Warning: Could not register special symbol {symbol_name}: {e}")


# Run the API symbols setup if the API module was loaded
if api_module_loaded:
    setup_api_symbols()

# We no longer need to skip test collection since we've fixed the imports
# The pytest_ignore_collect function has been removed


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--run-perf-tests",
        action="store_true",
        default=False,
        help="Run performance tests that are skipped by default",
    )
    parser.addoption(
        "--run-all-tests",
        action="store_true",
        default=False,
        help="Run all tests including skipped tests (except those requiring API keys)",
    )
    parser.addoption(
        "--run-api-tests",
        action="store_true",
        default=False,
        help="Run tests that require API keys and external services",
    )


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
