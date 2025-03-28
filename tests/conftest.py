"""Configure pytest environment for unit tests."""

import logging
import sys
from pathlib import Path

import pytest

# Get absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SRC_PATH = PROJECT_ROOT / "src"

logger = logging.getLogger(__name__)

# Add src directory first for proper imports
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure that the root conftest.py has run
if "ember" not in sys.modules:
    # Try to import it directly to get its side effects
    try:
        from ember import initialize_ember

        logger.info("Using root conftest.py's ember import")
    except ImportError:
        logger.warning("Could not import initialize_ember from root conftest.py")

        # Create a minimal fallback
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

            logger.info("Using fallback initialize_ember implementation")
            return ModelRegistry()

        # Create a minimal module
        import types

        ember_module = types.ModuleType("ember")
        ember_module.initialize_ember = fallback_initialize_ember
        sys.modules["ember"] = ember_module
        logger.info("Created fallback ember module")

# Suppress irrelevant warnings during tests
import warnings

warnings.filterwarnings("ignore", message=".*XCS functionality partially unavailable.*")
warnings.filterwarnings("ignore", message=".*initialize_ember\\(\\) is deprecated.*")

from typing import Any

from ember.core.registry.model.base.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)

# Register test providers for unit and integration tests
from ember.core.registry.model.providers.base_provider import BaseProviderModel
from ember.plugin_system import registered_providers


# Test Provider for test_model_registry
class TestProvider(BaseProviderModel):
    """Test provider for unit tests."""

    PROVIDER_NAME = "TestProvider"

    def create_client(self) -> Any:
        """Return a simple mock client."""
        return self

    def forward(self, request: ChatRequest) -> ChatResponse:
        """Process the request and return a response."""
        return ChatResponse(data=request.prompt.upper())


# Test provider for service tests
class DummyServiceProvider(BaseProviderModel):
    """Test provider for service tests."""

    PROVIDER_NAME = "DummyService"

    def create_client(self) -> Any:
        """Return a simple mock client."""
        return self

    def forward(self, request: ChatRequest) -> ChatResponse:
        """Process the request and return a response."""
        return ChatResponse(data=f"Echo: {request.prompt}")


# Async test provider
class DummyAsyncProvider(BaseProviderModel):
    """Async test provider for service tests."""

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


# Error test provider
class DummyErrorProvider(BaseProviderModel):
    """Provider that raises errors for testing."""

    PROVIDER_NAME = "DummyErrorService"

    def create_client(self) -> Any:
        """Return a simple mock client."""
        return self

    def forward(self, request: ChatRequest) -> ChatResponse:
        """Always raise an error when called."""
        raise RuntimeError(f"Async error invoking model {self.model_info.id}")


# Failing provider for integration tests
class FailingProvider(BaseProviderModel):
    """Provider that simulates failures for integration tests."""

    PROVIDER_NAME = "FailingProvider"

    def create_client(self) -> Any:
        """Return a simple mock client."""
        return self

    def forward(self, request: ChatRequest) -> ChatResponse:
        """Simulate a provider failure."""
        raise RuntimeError("Simulated provider API failure")


# Register all test providers
registered_providers["TestProvider"] = TestProvider
registered_providers["DummyService"] = DummyServiceProvider
registered_providers["DummyAsyncService"] = DummyAsyncProvider
registered_providers["DummyErrorService"] = DummyErrorProvider
registered_providers["FailingProvider"] = FailingProvider

# Enable testing mode for ModelFactory to use our registered providers
from ember.core.registry.model.base.registry.factory import ModelFactory

ModelFactory.enable_testing_mode()


@pytest.fixture(scope="session", autouse=True)
def global_setup_teardown():
    """
    Global fixture for session-level setup/teardown.
    - Can configure logging or environment variables here.
    """
    # Setup phase
    yield
    # Teardown phase


@pytest.fixture
def mock_lm_generation(mocker):
    """
    Mocks LMModule model_instance.generate calls to return predictable responses.
    Ensures deterministic tests regardless of input prompt.
    """

    def mock_generate(prompt, temperature=1.0, max_tokens=None):
        return f"Mocked response: {prompt}, temp={temperature}"

    # Patch the DummyModel generate method.
    mocker.patch(
        "tests.get_model_registry().DummyModel.generate", side_effect=mock_generate
    )
