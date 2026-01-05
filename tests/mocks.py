"""Test mocks."""

import hashlib
from typing import Any, Dict, Iterator, List, Optional

from ember.api.data import DataSource
from ember.models.providers.base import BaseProvider
from ember.models.schemas import ChatResponse, UsageStats


class MockProvider(BaseProvider):
    """Deterministic mock provider for testing.

    - No API keys required
    - Deterministic responses based on input hash
    - Tracks call history for assertions
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None, seed: int = 42):
        """Initialize mock provider.

        Args:
            responses: Map of input patterns to responses
            seed: Random seed for deterministic behavior
        """
        self.responses = responses or self._default_responses()
        self.seed = seed
        self.call_history = []
        self._call_count = 0

    def _default_responses(self) -> Dict[str, str]:
        """Default responses for common test cases."""
        return {
            "What is 2+2?": "4",
            "default": "Test response",
            "error": "Error: Invalid input",
            "hello": "Hello! How can I help you?",
            "summarize": "Summary: This is a test summary.",
        }

    def _hash_input(self, text: str) -> str:
        """Generate deterministic hash of input."""
        return hashlib.md5(f"{text}{self.seed}".encode()).hexdigest()

    def complete(
        self, messages: List[Dict[str, Any]], model: Optional[str] = None, **kwargs
    ) -> ChatResponse:
        """Generate deterministic completion."""
        self._call_count += 1

        # Extract text from messages
        if messages:
            text = messages[-1].get("content", "")
        else:
            text = "empty"

        # Record call
        self.call_history.append(
            {
                "messages": messages,
                "model": model,
                "kwargs": kwargs,
                "call_number": self._call_count,
            }
        )

        # Find response
        response_text = None
        for pattern, response in self.responses.items():
            if pattern in text:
                response_text = response
                break

        if response_text is None:
            # Generate deterministic response based on hash
            response_text = f"Response_{self._hash_input(text)[:8]}"

        return ChatResponse(
            data=response_text,
            model_id=model or "mock-model",
            usage=UsageStats(
                prompt_tokens=len(text.split()),
                completion_tokens=len(response_text.split()),
                cost_usd=0.0,
            ),
        )

    def stream(
        self, messages: List[Dict[str, Any]], model: Optional[str] = None, **kwargs
    ) -> Iterator[str]:
        """Stream response token by token."""
        response = self.complete(messages, model, **kwargs)

        # Simulate streaming by yielding words
        for word in response.data.split():
            yield word + " "


class MockDataSource(DataSource):
    """In-memory data source for testing."""

    def __init__(self, data: List[Dict[str, Any]], batch_size: int = 32):
        """Initialize with test data.

        Args:
            data: List of data items
            batch_size: Default batch size
        """
        self.data = data
        self._batch_size = batch_size

    def read_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
        """Yield data in batches."""
        batch_size = batch_size or self._batch_size

        for i in range(0, len(self.data), batch_size):
            yield self.data[i : i + batch_size]


class DeterministicMockLLM:
    """Simplified LLM mock for operator tests."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        """Initialize with response mapping."""
        self.responses = responses or {
            "default": "Mock LLM response",
            "test": "Test successful",
        }
        self.call_count = 0
        self.last_input = None

    def __call__(self, prompt: str, **kwargs) -> str:
        """Generate response for prompt."""
        self.call_count += 1
        self.last_input = prompt

        # Check for exact matches first
        if prompt in self.responses:
            return self.responses[prompt]

        # Check for partial matches
        for key, response in self.responses.items():
            if key in prompt:
                return response

        # Default response
        return self.responses.get("default", f"Response for: {prompt[:50]}...")


class ErrorMockProvider(BaseProvider):
    """Provider that always raises errors for testing error handling."""

    def __init__(self, error_type: type = RuntimeError, error_msg: str = "Mock error"):
        self.error_type = error_type
        self.error_msg = error_msg
        self.call_count = 0

    def complete(self, messages: List[Dict[str, Any]], **kwargs) -> ChatResponse:
        """Always raise an error."""
        self.call_count += 1
        raise self.error_type(self.error_msg)


class LatencyMockProvider(BaseProvider):
    """Provider with configurable latency for performance testing."""

    def __init__(self, latency_ms: float = 100, base_provider: Optional[BaseProvider] = None):
        """Initialize with latency configuration.

        Args:
            latency_ms: Milliseconds to delay
            base_provider: Provider to wrap (defaults to MockProvider)
        """
        self.latency_ms = latency_ms
        self.base_provider = base_provider or MockProvider()

    def complete(self, messages: List[Dict[str, Any]], **kwargs) -> ChatResponse:
        """Complete with added latency."""
        import time

        time.sleep(self.latency_ms / 1000.0)
        return self.base_provider.complete(messages, **kwargs)


# Golden response fixtures
GOLDEN_RESPONSES = {
    "math": {
        "What is 2+2?": "4",
        "What is 10*10?": "100",
        "Solve x^2 = 4": "x = 2 or x = -2",
    },
    "coding": {
        "Write hello world in Python": 'print("Hello, World!")',
        "What is a list comprehension?": "A list comprehension is a concise way to "
        "create lists in Python.",
    },
    "general": {
        "Hello": "Hello! How can I help you today?",
        "Goodbye": "Goodbye! Have a great day!",
    },
}


def create_golden_provider(category: str = "general") -> MockProvider:
    """Create a provider with golden responses for a category."""
    responses = GOLDEN_RESPONSES.get(category, GOLDEN_RESPONSES["general"])
    return MockProvider(responses=responses)
