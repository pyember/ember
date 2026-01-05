"""Reusable test fixtures."""

from datetime import UTC, datetime
from typing import Any, List

import pytest

from tests.test_constants import APIKeys, Models, TestData
from tests.test_doubles import (
    ChatResponse,
    FakeContext,
    FakeDataSource,
    FakeModelRegistry,
    FakeProvider,
    UsageStats,
    create_registry_with_models,
)


# Provider fixtures
@pytest.fixture
def fake_provider():
    """Basic fake provider for testing."""
    return FakeProvider(
        responses={
            TestData.SIMPLE_PROMPT: TestData.SIMPLE_RESPONSE,
            "Hello": "Hi there!",
            "Test": "Test response",
        }
    )


@pytest.fixture
def fake_provider_factory():
    """Factory for creating customized fake providers."""

    def _create(responses=None, should_fail=False, latency_ms=0):
        return FakeProvider(
            responses=responses or {}, should_fail=should_fail, latency_ms=latency_ms
        )

    return _create


# Registry fixtures
@pytest.fixture
def fake_registry():
    """Registry with standard test models pre-registered."""
    return create_registry_with_models()


@pytest.fixture
def empty_registry():
    """Empty registry for testing registration."""
    return FakeModelRegistry()


# Context fixtures
@pytest.fixture
def isolated_context(tmp_path, monkeypatch):
    """Completely isolated context for testing.

    This fixture ensures:
    - Clean temporary directory
    - No environment variables
    - Fresh context instance
    - Proper cleanup
    """
    # Create isolated home
    home = tmp_path / "test_home"
    home.mkdir()
    ember_dir = home / ".ember"
    ember_dir.mkdir()

    # Clear environment
    env_vars = [APIKeys.ENV_OPENAI, APIKeys.ENV_ANTHROPIC, APIKeys.ENV_GOOGLE]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)

    # Set test environment
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("EMBER_HOME", str(ember_dir))

    # Create fake context
    ctx = FakeContext(isolated=True)

    yield ctx

    # Cleanup is automatic with tmp_path


@pytest.fixture
def context_with_config(isolated_context):
    """Context with some pre-configured values."""
    ctx = isolated_context

    # Add some default config
    ctx.set_config("model", Models.GPT4)
    ctx.set_config("provider", "openai")
    ctx.set_config("api_keys.openai", APIKeys.OPENAI)
    ctx.set_config("timeouts.api", 5.0)

    return ctx


# Response fixtures
@pytest.fixture
def api_response_factory():
    """Factory for creating API responses."""

    class ResponseFactory:
        @staticmethod
        def create(
            content: str = "Test response",
            model: str = Models.GPT4,
            prompt_tokens: int = 10,
            completion_tokens: int = 20,
        ) -> ChatResponse:
            # Use real cost calculation from ember.models.pricing
            from ember.models.pricing import get_model_cost

            cost_info = get_model_cost(model)
            if cost_info:
                input_cost = (prompt_tokens / 1_000_000.0) * cost_info.get("input", 0.0)
                output_cost = (completion_tokens / 1_000_000.0) * cost_info.get("output", 0.0)
                cost_usd = round(input_cost + output_cost, 6)
            else:
                # Fallback for unknown models
                cost_usd = 0.0

            response = ChatResponse(
                data=content,
                model_id=model,
                usage=UsageStats(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    cost_usd=cost_usd,
                    actual_cost_usd=None,
                ),
            )

            now = datetime.now(UTC)
            response.created_at = now
            response.started_at = now
            response.completed_at = now
            response.latency_ms = 42.0
            return response

        @staticmethod
        def create_error_response(error: str = "API Error") -> ChatResponse:
            response = ChatResponse(
                data=f"Error: {error}",
                model_id="error",
                usage=UsageStats(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    cost_usd=0,
                    actual_cost_usd=None,
                ),
            )
            now = datetime.now(UTC)
            response.created_at = now
            response.started_at = now
            response.completed_at = now
            response.latency_ms = 0.0
            return response

    return ResponseFactory()


# Simple function version for direct use
def create_api_response(
    content: str = "Test response",
    model: str = Models.GPT4,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> ChatResponse:
    """Create API response directly without fixture complexity."""
    # Use real cost calculation from ember.models.pricing
    from ember.models.pricing import get_model_cost

    cost_info = get_model_cost(model)
    if cost_info:
        input_cost = (prompt_tokens / 1_000_000.0) * cost_info.get("input", 0.0)
        output_cost = (completion_tokens / 1_000_000.0) * cost_info.get("output", 0.0)
        cost_usd = round(input_cost + output_cost, 6)
    else:
        # Fallback for unknown models
        cost_usd = 0.0

    response = ChatResponse(
        data=content,
        model_id=model,
        usage=UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
            actual_cost_usd=None,  # Add the new field
        ),
    )
    now = datetime.now(UTC)
    response.created_at = now
    response.started_at = now
    response.completed_at = now
    response.latency_ms = 42.0
    return response


# Data fixtures
@pytest.fixture
def sample_data():
    """Sample data for testing data operations."""
    return TestData.SAMPLE_JSON_DATA.copy()


@pytest.fixture
def fake_data_source(sample_data):
    """Fake data source with sample data."""
    return FakeDataSource(sample_data)


# File fixtures
@pytest.fixture
def temp_json_file(tmp_path, sample_data):
    """Create temporary JSON file with sample data."""
    import json

    file_path = tmp_path / "test_data.json"
    file_path.write_text(json.dumps(sample_data, indent=2))
    return file_path


@pytest.fixture
def temp_yaml_file(tmp_path):
    """Create temporary YAML file."""
    import yaml

    data = {
        "model": Models.GPT4,
        "provider": "openai",
        "config": {"temperature": 0.7, "max_tokens": 100},
    }

    file_path = tmp_path / "test_config.yaml"
    file_path.write_text(yaml.dump(data))
    return file_path


@pytest.fixture
def temp_csv_file(tmp_path):
    """Create temporary CSV file."""
    csv_content = """id,text,label
1,First item,A
2,Second item,B
3,Third item,A"""

    file_path = tmp_path / "test_data.csv"
    file_path.write_text(csv_content)
    return file_path


# Environment fixtures
@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment without API keys."""
    env_vars = [
        APIKeys.ENV_OPENAI,
        APIKeys.ENV_ANTHROPIC,
        APIKeys.ENV_GOOGLE,
        "EMBER_ENV",
        "EMBER_HOME",
    ]

    for var in env_vars:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def test_api_keys(monkeypatch):
    """Set test API keys in environment."""
    monkeypatch.setenv(APIKeys.ENV_OPENAI, APIKeys.OPENAI)
    monkeypatch.setenv(APIKeys.ENV_ANTHROPIC, APIKeys.ANTHROPIC)
    monkeypatch.setenv(APIKeys.ENV_GOOGLE, APIKeys.GOOGLE)


# Performance fixtures
@pytest.fixture
def benchmark_timer():
    """Simple timer for performance measurements."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start_time

        def assert_faster_than(self, seconds: float):
            assert self.elapsed < seconds, f"Too slow: {self.elapsed:.3f}s > {seconds}s"

        def assert_slower_than(self, seconds: float):
            assert self.elapsed > seconds, f"Too fast: {self.elapsed:.3f}s < {seconds}s"

    return Timer()


# Assertion helpers as fixtures
@pytest.fixture
def assert_error_matches():
    """Helper for asserting error patterns."""

    def _assert(exc_info, pattern):
        """Assert that exception matches pattern."""
        from tests.test_constants import ErrorPatterns

        error_str = str(exc_info.value)

        # If pattern is a string, get it from ErrorPatterns
        if isinstance(pattern, str):
            pattern = getattr(ErrorPatterns, pattern)

        assert pattern.search(error_str), f"Error '{error_str}' doesn't match pattern"

    return _assert


@pytest.fixture
def assert_response_valid():
    """Helper for validating API responses."""

    def _assert(response: ChatResponse):
        """Assert that response is valid."""
        assert response is not None
        assert response.data
        assert response.model_id
        assert response.usage.total_tokens >= 0
        assert response.usage.cost_usd >= 0

    return _assert


# Batch operation fixtures
@pytest.fixture
def batch_processor():
    """Simple batch processor for testing."""

    def process_batch(items: List[Any], batch_size: int = 32):
        """Process items in batches."""
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            # Simple processing - just uppercase strings
            batch_results = [item.upper() if isinstance(item, str) else item for item in batch]
            results.extend(batch_results)
        return results

    return process_batch
