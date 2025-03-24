"""
Root conftest.py for pytest configuration
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

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

# Direct import from src.ember
try:
    # Import src.ember module
    src_ember = importlib.import_module("src.ember")

    # Make it available as "ember"
    sys.modules["ember"] = src_ember
    print("Successfully mapped src.ember -> ember")

    # Ensure initialize_ember is available
    if hasattr(src_ember, "initialize_ember"):
        print("Found initialize_ember in ember module")
    else:
        # Try to directly import it
        try:
            from src.ember import initialize_ember

            src_ember.initialize_ember = initialize_ember
            print("Added initialize_ember to ember module")
        except ImportError as e:
            print(f"Warning: Could not import initialize_ember: {e}")

            # Create a stub implementation
            def fallback_initialize_ember(
                config_path=None,
                auto_discover=True,
                force_discovery=False,
                api_keys=None,
                env_prefix="EMBER_",
                initialize_context=True,
            ):
                """Fallback implementation of initialize_ember for testing."""
                from src.ember.core.registry.model.base.registry.model_registry import (
                    ModelRegistry,
                )

                print("Using fallback initialize_ember implementation")
                return ModelRegistry()

            src_ember.initialize_ember = fallback_initialize_ember
            print("Created fallback initialize_ember function")
except ImportError as e:
    print(f"Error importing src.ember: {e}")

# Setup module aliases for critical Ember components
for base_module in ["core", "xcs", "api"]:
    module_path = f"src.ember.{base_module}"
    target_path = f"ember.{base_module}"
    try:
        module = importlib.import_module(module_path)
        sys.modules[target_path] = module
        print(f"Mapped {module_path} -> {target_path}")
    except ImportError as e:
        print(f"Warning: Could not import {module_path}: {e}")

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
        metadata_registry,
        registry,
    )

    sys.modules["ember.core.utils.data.registry"] = registry
    sys.modules["ember.core.utils.data.metadata_registry"] = metadata_registry
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
        commonsense_qa,
        halueval,
        mmlu,
        short_answer,
        truthful_qa,
    )

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
