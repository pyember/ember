"""Shared constants used across tests."""

import re
from typing import Final


class Models:
    """Standard model identifiers for testing."""

    # OpenAI models
    GPT4: Final[str] = "gpt-4"
    GPT4_TURBO: Final[str] = "gpt-4-turbo"
    GPT35: Final[str] = "gpt-3.5-turbo"

    # Anthropic models
    CLAUDE3: Final[str] = "claude-3-opus"
    CLAUDE3_SONNET: Final[str] = "claude-3-sonnet"

    # Google models
    GEMINI_PRO: Final[str] = "gemini-1.5-pro-latest"

    # Invalid model for error testing
    INVALID: Final[str] = "invalid-model-xyz"

    # Lists for parameterized testing
    ALL_VALID: Final[list[str]] = [GPT4, GPT35, CLAUDE3, GEMINI_PRO]
    OPENAI_MODELS: Final[list[str]] = [GPT4, GPT4_TURBO, GPT35]
    ANTHROPIC_MODELS: Final[list[str]] = [CLAUDE3, CLAUDE3_SONNET]


class APIKeys:
    """Test API keys - clearly fake to avoid accidents."""

    OPENAI: Final[str] = "sk-test-1234567890abcdef"
    ANTHROPIC: Final[str] = "ant-test-1234567890abcdef"
    GOOGLE: Final[str] = "goog-test-1234567890abcdef"

    # Environment variable names
    ENV_OPENAI: Final[str] = "OPENAI_API_KEY"
    ENV_ANTHROPIC: Final[str] = "ANTHROPIC_API_KEY"
    ENV_GOOGLE: Final[str] = "GOOGLE_API_KEY"


class ErrorPatterns:
    """Regex patterns for error matching - precompiled for performance."""

    MISSING_API_KEY: Final[re.Pattern] = re.compile(
        r"(?i)(no api key|api key.*not found|missing.*api key|api key.*required)"
    )
    INVALID_MODEL: Final[re.Pattern] = re.compile(
        r"(?i)(unknown model|model.*not found|invalid model|cannot determine provider)"
    )
    RATE_LIMIT: Final[re.Pattern] = re.compile(r"(?i)(rate limit|too many requests|quota exceeded)")
    CONNECTION_ERROR: Final[re.Pattern] = re.compile(
        r"(?i)(connection error|network error|timeout|cannot connect)"
    )
    PERMISSION_DENIED: Final[re.Pattern] = re.compile(
        r"(?i)(permission denied|access denied|unauthorized|forbidden)"
    )


class TestData:
    """Common test data to avoid duplication."""

    # Sample prompts
    SIMPLE_PROMPT: Final[str] = "Hello, world!"
    EMPTY_PROMPT: Final[str] = ""
    LONG_PROMPT: Final[str] = "Tell me a story. " * 100  # ~400 tokens

    # Sample responses
    SIMPLE_RESPONSE: Final[str] = "Hello! How can I help you today?"
    ERROR_RESPONSE: Final[str] = "An error occurred"

    # Test messages for chat format
    SIMPLE_MESSAGES: Final[list[dict]] = [{"role": "user", "content": "Hello"}]

    # Test JSON data
    SAMPLE_JSON_DATA: Final[list[dict]] = [
        {"id": 1, "text": "First item", "label": "A"},
        {"id": 2, "text": "Second item", "label": "B"},
        {"id": 3, "text": "Third item", "label": "A"},
    ]


class Timeouts:
    """Test timeout constants - short for fast feedback."""

    UNIT_TEST: Final[float] = 1.0  # 1 second for unit tests
    INTEGRATION_TEST: Final[float] = 10.0  # 10 seconds for integration
    E2E_TEST: Final[float] = 30.0  # 30 seconds for end-to-end
    BENCHMARK: Final[float] = 5.0  # 5 seconds for benchmarks

    # Network timeouts
    API_CALL: Final[float] = 5.0  # 5 seconds for API calls
    FILE_OPERATION: Final[float] = 2.0  # 2 seconds for file ops

    # Performance test thresholds
    FAST_OPERATION: Final[float] = 0.01  # 10ms for fast operations
    MEDIUM_OPERATION: Final[float] = 0.1  # 100ms for medium operations
    SLOW_OPERATION: Final[float] = 1.0  # 1s for slow operations


class Paths:
    """Common test paths - relative to test directory."""

    TEST_DATA_DIR: Final[str] = "test_data"
    FIXTURES_DIR: Final[str] = "fixtures"
    SNAPSHOTS_DIR: Final[str] = "snapshots"

    # File patterns
    JSON_PATTERN: Final[str] = "*.json"
    YAML_PATTERN: Final[str] = "*.yaml"
    PY_PATTERN: Final[str] = "*.py"


class Performance:
    """Performance thresholds for tests."""

    # Operations per second thresholds
    CONTEXT_OPS_PER_SEC: Final[int] = 100_000
    CONFIG_OPS_PER_SEC: Final[int] = 50_000
    MODEL_CREATE_PER_SEC: Final[int] = 1_000

    # Relative performance ratios
    SLOWDOWN_TOLERANCE: Final[float] = 1.5  # Allow 50% slowdown
    SPEEDUP_REQUIRED: Final[float] = 0.8  # Require 20% speedup


# Validation helpers
def is_valid_api_key(key: str) -> bool:
    """Check if an API key looks valid (for testing)."""
    return bool(key and len(key) > 10 and key.startswith(("sk-", "ant-", "goog-")))


def is_test_api_key(key: str) -> bool:
    """Check if this is one of our test API keys."""
    return key in {APIKeys.OPENAI, APIKeys.ANTHROPIC, APIKeys.GOOGLE}
