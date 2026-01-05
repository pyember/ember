"""Pytest configuration and fixtures."""

import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

try:
    from .test_constants import Models, TestData
    from .test_doubles import FakeContext, FakeModelRegistry

    pytest_plugins = ("tests.fixtures",)
except ImportError:
    from test_constants import Models, TestData
    from test_doubles import FakeContext, FakeModelRegistry

    pytest_plugins = ("fixtures",)

# Optional provider stubs to avoid heavy runtime dependencies during unit tests.


@pytest.fixture(scope="session")
def anyio_backend():
    """Force AnyIO-powered tests to run on asyncio backend only."""

    return "asyncio"


def _install_stub(fullname: str, attrs: dict | None = None) -> None:
    parts = fullname.split(".")
    module: types.ModuleType | None = None
    qualname = ""
    for part in parts:
        qualname = f"{qualname}.{part}" if qualname else part
        existing = sys.modules.get(qualname)
        if existing is None:
            new_module = types.ModuleType(qualname)
            sys.modules[qualname] = new_module
            if module is not None:
                setattr(module, part, new_module)
            module = new_module
        else:
            module = existing
    if attrs and module is not None:
        for key, value in attrs.items():
            setattr(module, key, value)


if "anthropic" not in sys.modules:

    class _AnthropicError(RuntimeError):
        pass

    class _Dummy:
        pass

    class _AnthropicClient:
        def __init__(self, api_key: str | None = None):
            self.api_key = api_key
            self.messages = types.SimpleNamespace(create=self._create)

        def _create(self, **kwargs):  # pragma: no cover
            return types.SimpleNamespace(
                content=[],
                usage=types.SimpleNamespace(input_tokens=0, output_tokens=0),
            )

    _install_stub(
        "anthropic",
        attrs={
            "Anthropic": _AnthropicClient,
            "AuthenticationError": _AnthropicError,
            "RateLimitError": _AnthropicError,
            "APIError": _AnthropicError,
        },
    )
    anthropic_module = sys.modules.get("anthropic")
    if anthropic_module is not None:
        anthropic_module.__path__ = ["."]

    _install_stub(
        "anthropic.types",
        attrs={
            "MessageParam": dict,
            "Message": dict,
            "ContentBlock": _Dummy,
            "TextBlock": _Dummy,
            "ThinkingBlock": _Dummy,
            "RedactedThinkingBlock": _Dummy,
            "ToolUseBlock": _Dummy,
            "ServerToolUseBlock": _Dummy,
            "WebSearchToolResultBlock": _Dummy,
        },
    )

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Configure pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


# Core test fixtures
@pytest.fixture
def tmp_ctx(tmp_path, monkeypatch):
    """Isolated EmberContext with temporary home directory.

    Provides complete isolation from user's real ~/.ember directory.
    All tests using this fixture are hermetic and parallelizable.
    """
    # Import only if we need the real context
    try:
        from ember._internal.context import EmberContext

        # Create fake home
        home = tmp_path / "home"
        home.mkdir()
        ember_dir = home / ".ember"
        ember_dir.mkdir()

        # Set environment to use temp directory
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("EMBER_HOME", str(ember_dir))
        monkeypatch.delenv("EMBER_CONFIG_PATH", raising=False)

        # Try to use public API for context reset if available
        if hasattr(EmberContext, "reset"):
            EmberContext.reset()

        # Create fresh context
        ctx = EmberContext(isolated=True)
        # Promote to current context so code paths using EmberContext.current()
        # during the test see this isolated configuration.
        EmberContext._thread_local.context = ctx
        EmberContext._context_var.set(ctx)
        yield ctx

        # Cleanup via public API if available
        if hasattr(EmberContext, "reset"):
            EmberContext.reset()

    except ImportError:
        # If real context not available, use fake
        ctx = FakeContext(isolated=True)
        yield ctx


@pytest.fixture
def mock_cli_args(monkeypatch):
    """Mock sys.argv for CLI testing without subprocess."""

    def _mock_args(*args):
        monkeypatch.setattr(sys, "argv", ["ember"] + list(args))

    return _mock_args


# Model API fixtures
@pytest.fixture
def mock_model_response():
    """Standard mock response for model tests."""
    from tests.fixtures import create_api_response

    return create_api_response(
        content="Test response", model=Models.GPT4, prompt_tokens=10, completion_tokens=20
    )


@pytest.fixture
def mock_registry():
    """Mock model registry for testing."""
    return FakeModelRegistry()


# Data API fixtures
@pytest.fixture
def temp_data_file(tmp_path):
    """Create temporary JSON data file."""
    data = TestData.SAMPLE_JSON_DATA.copy()
    file_path = tmp_path / "test_data.json"
    file_path.write_text(json.dumps(data))
    return file_path


@pytest.fixture
def temp_csv_file(tmp_path):
    """Create temporary CSV data file."""
    rows = ["text,label"]
    for item in TestData.SAMPLE_JSON_DATA:
        rows.append(f'{item["text"]},{item["label"]}')
    csv_content = "\n".join(rows)
    file_path = tmp_path / "test_data.csv"
    file_path.write_text(csv_content)
    return file_path


# Operator fixtures
@pytest.fixture
def simple_operator():
    """Simple operator for testing."""

    def double(x):
        return x * 2

    return double


@pytest.fixture
def mock_model_operator():
    """Mock model operator for testing."""

    def model_op(text):
        return Mock(text=f"Processed: {text}", usage={"tokens": 10})

    return model_op


# XCS fixtures
@pytest.fixture
def slow_function():
    """Function that simulates slow operation."""
    import time

    def slow_op(x):
        time.sleep(0.1)
        return x

    return slow_op
