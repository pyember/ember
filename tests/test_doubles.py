"""Test doubles for unit tests."""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

try:
    from .test_constants import Models, TestData
except ImportError:
    from test_constants import Models, TestData


class ModelProvider(Protocol):
    """Protocol defining the provider interface."""

    def complete(self, prompt: str, model: str, **kwargs) -> "ChatResponse":
        """Complete a prompt with the model."""
        ...


@dataclass
class ChatResponse:
    """Minimal response object for testing."""

    data: str
    model_id: str
    usage: "UsageStats"

    @property
    def text(self) -> str:
        return self.data


@dataclass
class UsageStats:
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    actual_cost_usd: Optional[float] = None


@dataclass
class FakeProvider:
    """Fake provider with deterministic responses.

    This replaces complex mocking with simple, predictable behavior.
    """

    responses: Dict[str, str] = field(default_factory=dict)
    call_history: List[Dict] = field(default_factory=list)
    latency_ms: float = 0
    should_fail: bool = False
    failure_message: str = "Fake provider error"
    api_key: Optional[str] = None

    def __post_init__(self):
        """Initialize with default responses if none provided."""
        if not self.responses:
            self.responses = {
                TestData.SIMPLE_PROMPT: TestData.SIMPLE_RESPONSE,
                "Hello": "Hi there!",
                "Test": "Test response",
            }

    def complete(self, prompt: str, model: str, **kwargs) -> ChatResponse:
        """Complete with deterministic response."""
        # Record the call
        self.call_history.append(
            {"prompt": prompt, "model": model, "kwargs": kwargs, "timestamp": time.time()}
        )

        # Simulate latency if configured
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000)

        # Fail if configured to
        if self.should_fail:
            raise Exception(self.failure_message)

        # Deterministic response based on prompt
        response_text = self.responses.get(prompt, f"Response to: {prompt[:50]}")

        # Calculate tokens simply (word count * 1.3)
        prompt_tokens = int(len(prompt.split()) * 1.3)
        completion_tokens = int(len(response_text.split()) * 1.3)

        return ChatResponse(
            data=response_text,
            model_id=model,
            usage=UsageStats(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=0.0001 * (prompt_tokens + completion_tokens),
                actual_cost_usd=None,
            ),
        )

    def reset(self):
        """Reset call history for fresh test."""
        self.call_history.clear()


class FakeModelRegistry:
    """Fake registry that behaves like real one but simpler."""

    def __init__(self, providers: Optional[Dict[str, ModelProvider]] = None):
        self.providers = providers or {}
        self.models: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._instances: Dict[str, Any] = {}

    def register_model(self, model_id: str, provider_name: str):
        """Register a model with a provider."""
        with self._lock:
            if provider_name not in self.providers:
                raise ValueError(f"Unknown provider: {provider_name}")
            self.models[model_id] = provider_name

    def get_model(self, model_id: str) -> ModelProvider:
        """Get a model by ID, with caching behavior."""
        with self._lock:
            # Check if model exists
            if model_id not in self.models:
                raise ValueError(f"Model not found: {model_id}")

            # Return cached instance if exists
            if model_id in self._instances:
                return self._instances[model_id]

            # Create and cache new instance
            provider_name = self.models[model_id]
            # Create a new provider instance each time to simulate real behavior
            provider_class = self.providers[provider_name]
            if isinstance(provider_class, type):
                provider = provider_class()
            else:
                # If it's already an instance, we need to create a new one
                provider = FakeProvider()
            self._instances[model_id] = provider
            return provider

    def list_models(self) -> List[str]:
        """List all registered models."""
        with self._lock:
            return list(self.models.keys())

    def clear_cache(self):
        """Clear the instance cache."""
        with self._lock:
            self._instances.clear()


class FakeContext:
    """Minimal fake for EmberContext."""

    def __init__(self, isolated: bool = True):
        self.isolated = isolated
        self._config: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def get_config(self, path: str, default: Any = None) -> Any:
        """Get config value by dot-separated path."""
        with self._lock:
            parts = path.split(".")
            value = self._config

            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default

            return value

    def set_config(self, path: str, value: Any):
        """Set config value by dot-separated path."""
        with self._lock:
            parts = path.split(".")
            config = self._config

            # Navigate to parent
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]

            # Set value
            config[parts[-1]] = value

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration."""
        with self._lock:
            return self._config.copy()


class FakeDataSource:
    """Minimal data source for testing data operations."""

    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items
        self.read_count = 0

    def read_batches(self, batch_size: int = 32):
        """Yield items in batches."""
        self.read_count += 1

        for i in range(0, len(self.items), batch_size):
            yield self.items[i : i + batch_size]


# Factory functions for common test scenarios
def create_failing_provider(error_message: str = "API Error") -> FakeProvider:
    """Create a provider that always fails."""
    return FakeProvider(should_fail=True, failure_message=error_message)


def create_slow_provider(latency_ms: float = 100) -> FakeProvider:
    """Create a provider with simulated latency."""
    return FakeProvider(latency_ms=latency_ms)


def create_registry_with_models() -> FakeModelRegistry:
    """Create a registry with standard test models."""
    registry = FakeModelRegistry(
        {
            "openai": FakeProvider(),
            "anthropic": FakeProvider(),
            "google": FakeProvider(),
        }
    )

    # Register standard models
    registry.register_model(Models.GPT4, "openai")
    registry.register_model(Models.GPT35, "openai")
    registry.register_model(Models.CLAUDE3, "anthropic")
    registry.register_model(Models.GEMINI_PRO, "google")

    return registry
