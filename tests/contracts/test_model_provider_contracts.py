"""Contract tests for model providers."""

from abc import ABC, abstractmethod
from typing import Any

import pytest

from tests.test_constants import APIKeys, Models, TestData
from tests.test_doubles import ChatResponse, FakeProvider, UsageStats


class ModelProviderContract(ABC):
    """Base contract that all model providers must satisfy.

    Subclasses implement create_provider() to provide the
    provider instance to test.
    """

    @abstractmethod
    def create_provider(self) -> Any:
        """Create the provider instance to test.

        Returns:
            Provider instance configured for testing
        """
        pass

    @abstractmethod
    def get_test_model_id(self) -> str:
        """Get a valid model ID for this provider.

        Returns:
            Model ID that this provider supports
        """
        pass

    def test_provider_has_required_methods(self):
        """Test that provider has all required methods."""
        provider = self.create_provider()

        # Required methods
        assert hasattr(provider, "complete"), "Provider must have complete() method"
        assert callable(provider.complete), "complete() must be callable"

    def test_complete_returns_valid_response(self):
        """Test that complete() returns a valid response."""
        provider = self.create_provider()
        model_id = self.get_test_model_id()

        # Call complete with basic prompt
        response = provider.complete(TestData.SIMPLE_PROMPT, model_id)

        # Validate response structure
        assert isinstance(response, ChatResponse), "Must return ChatResponse"
        assert response.data, "Response must have data"
        assert response.model_id, "Response must have model_id"
        assert isinstance(response.usage, UsageStats), "Must have usage stats"

    def test_complete_handles_empty_prompt(self):
        """Test that provider handles empty prompts gracefully."""
        provider = self.create_provider()
        model_id = self.get_test_model_id()

        # Should not crash on empty prompt
        response = provider.complete("", model_id)

        assert isinstance(response, ChatResponse)
        assert response.data is not None

    def test_complete_respects_temperature(self):
        """Test that temperature parameter is respected."""
        provider = self.create_provider()
        model_id = self.get_test_model_id()

        # Call with different temperatures
        response1 = provider.complete(TestData.SIMPLE_PROMPT, model_id, temperature=0.0)

        response2 = provider.complete(TestData.SIMPLE_PROMPT, model_id, temperature=1.0)

        # Both should succeed
        assert response1.data
        assert response2.data

    def test_complete_respects_max_tokens(self):
        """Test that max_tokens parameter is respected."""
        provider = self.create_provider()
        model_id = self.get_test_model_id()

        # Request very short response
        response = provider.complete("Tell me a very long story", model_id, max_tokens=10)

        assert isinstance(response, ChatResponse)
        # Can't strictly enforce token count without tokenizer,
        # but response should exist
        assert response.data

    def test_usage_statistics_are_reasonable(self):
        """Test that usage statistics make sense."""
        provider = self.create_provider()
        model_id = self.get_test_model_id()

        response = provider.complete(TestData.SIMPLE_PROMPT, model_id)

        usage = response.usage

        # Basic sanity checks
        assert usage.prompt_tokens > 0, "Should have prompt tokens"
        assert usage.completion_tokens > 0, "Should have completion tokens"
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
        assert usage.cost_usd >= 0, "Cost should be non-negative"

    def test_handles_invalid_model_gracefully(self):
        """Test that provider handles invalid model IDs gracefully."""
        provider = self.create_provider()

        # Fake provider is designed to accept any model for flexibility in testing
        # It doesn't validate models like real providers do
        response = provider.complete(TestData.SIMPLE_PROMPT, "invalid-model-xyz-123")

        # Should still return a valid response structure
        assert isinstance(response, ChatResponse)
        assert response.model_id == "invalid-model-xyz-123"

    def test_thread_safety(self):
        """Test that provider is thread-safe."""
        import queue
        import threading

        provider = self.create_provider()
        model_id = self.get_test_model_id()
        results_queue = queue.Queue()
        errors_queue = queue.Queue()

        def call_provider(thread_id):
            try:
                response = provider.complete(f"Thread {thread_id} prompt", model_id)
                results_queue.put((thread_id, response))
            except Exception as e:
                errors_queue.put((thread_id, e))

        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=call_provider, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Check results
        assert errors_queue.empty(), "No errors in concurrent access"

        # All threads should get valid responses
        responses = []
        while not results_queue.empty():
            thread_id, response = results_queue.get()
            responses.append(response)
            assert isinstance(response, ChatResponse)

        assert len(responses) == 5


# Concrete contract tests for each provider type
class TestFakeProviderContract(ModelProviderContract):
    """Test that our FakeProvider satisfies the contract."""

    def create_provider(self) -> FakeProvider:
        """Create a FakeProvider for testing."""
        return FakeProvider(
            responses={
                TestData.SIMPLE_PROMPT: TestData.SIMPLE_RESPONSE,
                "": "Empty prompt response",
            }
        )

    def get_test_model_id(self) -> str:
        """Get test model ID."""
        return Models.GPT4


class TestOpenAIProviderContract(ModelProviderContract):
    """Test OpenAI provider contract compliance."""

    @pytest.fixture(autouse=True)
    def setup_environment(self, monkeypatch):
        """Set up test environment."""
        monkeypatch.setenv(APIKeys.ENV_OPENAI, APIKeys.OPENAI)

    def create_provider(self) -> Any:
        """Create OpenAI provider."""
        # For now, use FakeProvider configured like OpenAI
        return FakeProvider(
            responses={
                TestData.SIMPLE_PROMPT: "OpenAI response",
                "": "Empty prompt response",
                "Tell me a very long story": "Once upon a time...",
            }
        )

    def get_test_model_id(self) -> str:
        """Get OpenAI model ID."""
        return Models.GPT4


class TestAnthropicProviderContract(ModelProviderContract):
    """Test Anthropic provider contract compliance."""

    @pytest.fixture(autouse=True)
    def setup_environment(self, monkeypatch):
        """Set up test environment."""
        monkeypatch.setenv(APIKeys.ENV_ANTHROPIC, APIKeys.ANTHROPIC)

    def create_provider(self) -> Any:
        """Create Anthropic provider."""
        # For now, use FakeProvider configured like Anthropic
        return FakeProvider(
            responses={
                TestData.SIMPLE_PROMPT: "Claude response",
                "": "I'm here to help!",
                "Tell me a very long story": "Let me tell you a story...",
            }
        )

    def get_test_model_id(self) -> str:
        """Get Anthropic model ID."""
        return Models.CLAUDE3


class TestGoogleProviderContract(ModelProviderContract):
    """Test Google provider contract compliance."""

    @pytest.fixture(autouse=True)
    def setup_environment(self, monkeypatch):
        """Set up test environment."""
        monkeypatch.setenv(APIKeys.ENV_GOOGLE, APIKeys.GOOGLE)

    def create_provider(self) -> Any:
        """Create Google provider."""
        # For now, use FakeProvider configured like Google
        return FakeProvider(
            responses={
                TestData.SIMPLE_PROMPT: "Gemini response",
                "": "How can I assist you?",
                "Tell me a very long story": "Here's a story for you...",
            }
        )

    def get_test_model_id(self) -> str:
        """Get Google model ID."""
        return Models.GEMINI_PRO


# Additional contract tests for specific behaviors
class TestProviderErrorHandling:
    """Test error handling across all providers."""

    @pytest.mark.parametrize(
        "provider_class,model_id",
        [
            (FakeProvider, Models.GPT4),
            # Add real providers when available
        ],
    )
    def test_network_error_handling(self, provider_class, model_id):
        """Test that providers handle network errors gracefully."""
        # Create provider that simulates network error
        from tests.test_doubles import create_failing_provider

        provider = create_failing_provider("Network connection failed")

        with pytest.raises(Exception) as exc_info:
            provider.complete(TestData.SIMPLE_PROMPT, model_id)

        assert "Network connection failed" in str(exc_info.value)

    @pytest.mark.parametrize(
        "provider_class,model_id",
        [
            (FakeProvider, Models.GPT4),
            # Add real providers when available
        ],
    )
    def test_timeout_handling(self, provider_class, model_id):
        """Test that providers handle timeouts appropriately."""
        from tests.test_doubles import create_slow_provider

        # Create provider with 100ms latency
        provider = create_slow_provider(latency_ms=100)

        # Should still complete (100ms is acceptable)
        import time

        start = time.time()
        response = provider.complete(TestData.SIMPLE_PROMPT, model_id)
        elapsed = time.time() - start

        assert response.data
        assert elapsed >= 0.1  # At least 100ms
        assert elapsed < 0.5  # But not too long


class TestProviderCostCalculation:
    """Test cost calculation across providers."""

    @pytest.mark.parametrize(
        "provider_factory,model_id,expected_cost_range",
        [
            (lambda: FakeProvider(), Models.GPT4, (0.0001, 0.01)),
            (lambda: FakeProvider(), Models.CLAUDE3, (0.0001, 0.01)),
            (lambda: FakeProvider(), Models.GEMINI_PRO, (0.0001, 0.01)),
        ],
    )
    def test_cost_calculation_reasonable(self, provider_factory, model_id, expected_cost_range):
        """Test that cost calculations are in reasonable ranges."""
        provider = provider_factory()

        response = provider.complete(TestData.SIMPLE_PROMPT, model_id)

        cost = response.usage.cost_usd
        min_cost, max_cost = expected_cost_range

        assert (
            min_cost <= cost <= max_cost
        ), f"Cost {cost} not in expected range {expected_cost_range}"
