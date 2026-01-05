"""Tests for test doubles."""

import threading
import time

import pytest

from tests.test_constants import Models
from tests.test_doubles import (
    ChatResponse,
    FakeContext,
    FakeDataSource,
    FakeModelRegistry,
    FakeProvider,
    create_failing_provider,
    create_registry_with_models,
    create_slow_provider,
)


class TestFakeProvider:
    """Test the FakeProvider test double."""

    def test_basic_completion(self):
        """Test basic completion works."""
        provider = FakeProvider()
        response = provider.complete("Hello", Models.GPT4)

        assert isinstance(response, ChatResponse)
        assert response.data == "Hi there!"
        assert response.model_id == Models.GPT4
        assert response.usage.total_tokens > 0

    def test_call_history_recording(self):
        """Test that calls are recorded."""
        provider = FakeProvider()

        provider.complete("Test1", Models.GPT4, temperature=0.7)
        provider.complete("Test2", Models.CLAUDE3, max_tokens=100)

        assert len(provider.call_history) == 2
        assert provider.call_history[0]["prompt"] == "Test1"
        assert provider.call_history[0]["kwargs"]["temperature"] == 0.7
        assert provider.call_history[1]["model"] == Models.CLAUDE3

    def test_custom_responses(self):
        """Test custom response configuration."""
        provider = FakeProvider(responses={"custom": "custom response", "test": "test response"})

        assert provider.complete("custom", Models.GPT4).data == "custom response"
        assert provider.complete("test", Models.GPT4).data == "test response"
        assert provider.complete("unknown", Models.GPT4).data.startswith("Response to:")

    def test_failure_simulation(self):
        """Test failure simulation."""
        provider = create_failing_provider("Network error")

        with pytest.raises(Exception, match="Network error"):
            provider.complete("test", Models.GPT4)

    def test_latency_simulation(self):
        """Test latency simulation."""
        provider = create_slow_provider(latency_ms=50)

        start = time.time()
        provider.complete("test", Models.GPT4)
        elapsed = time.time() - start

        assert elapsed >= 0.05  # At least 50ms
        assert elapsed < 0.1  # But not too long

    def test_reset_functionality(self):
        """Test reset clears history."""
        provider = FakeProvider()

        provider.complete("test1", Models.GPT4)
        provider.complete("test2", Models.GPT4)
        assert len(provider.call_history) == 2

        provider.reset()
        assert len(provider.call_history) == 0


class TestFakeModelRegistry:
    """Test the FakeModelRegistry test double."""

    def test_model_registration(self):
        """Test registering and retrieving models."""
        registry = FakeModelRegistry({"openai": FakeProvider()})

        registry.register_model(Models.GPT4, "openai")

        model = registry.get_model(Models.GPT4)
        assert isinstance(model, FakeProvider)

    def test_unknown_provider_error(self):
        """Test error for unknown provider."""
        registry = FakeModelRegistry()

        with pytest.raises(ValueError, match="Unknown provider"):
            registry.register_model(Models.GPT4, "unknown")

    def test_unknown_model_error(self):
        """Test error for unknown model."""
        registry = FakeModelRegistry()

        with pytest.raises(ValueError, match="Model not found"):
            registry.get_model(Models.INVALID)

    def test_model_caching(self):
        """Test that models are cached."""
        registry = create_registry_with_models()

        model1 = registry.get_model(Models.GPT4)
        model2 = registry.get_model(Models.GPT4)

        assert model1 is model2  # Same instance

    def test_list_models(self):
        """Test listing registered models."""
        registry = create_registry_with_models()

        models = registry.list_models()
        assert Models.GPT4 in models
        assert Models.CLAUDE3 in models
        assert Models.GEMINI_PRO in models

    def test_clear_cache(self):
        """Test cache clearing."""
        registry = create_registry_with_models()

        model1 = registry.get_model(Models.GPT4)
        registry.clear_cache()
        model2 = registry.get_model(Models.GPT4)

        # Different instances after cache clear
        assert model1 is not model2

    def test_thread_safety(self):
        """Test thread-safe operations."""
        registry = create_registry_with_models()
        results = []
        errors = []

        def get_model():
            try:
                model = registry.get_model(Models.GPT4)
                results.append(model)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_model) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All should get same instance
        assert all(m is results[0] for m in results)


class TestFakeContext:
    """Test the FakeContext test double."""

    def test_simple_config_operations(self):
        """Test basic get/set operations."""
        ctx = FakeContext()

        ctx.set_config("key", "value")
        assert ctx.get_config("key") == "value"
        assert ctx.get_config("missing", "default") == "default"

    def test_nested_config_operations(self):
        """Test nested path operations."""
        ctx = FakeContext()

        ctx.set_config("a.b.c", "deep value")
        assert ctx.get_config("a.b.c") == "deep value"
        assert ctx.get_config("a.b") == {"c": "deep value"}
        assert ctx.get_config("a") == {"b": {"c": "deep value"}}

    def test_get_all_config(self):
        """Test getting all configuration."""
        ctx = FakeContext()

        ctx.set_config("key1", "value1")
        ctx.set_config("key2", "value2")

        all_config = ctx.get_all_config()
        assert all_config["key1"] == "value1"
        assert all_config["key2"] == "value2"

    def test_thread_safety(self):
        """Test thread-safe config access."""
        ctx = FakeContext()

        def writer(n):
            for i in range(10):
                ctx.set_config(f"thread{n}.item{i}", i)

        def reader(n):
            for i in range(10):
                ctx.get_config(f"thread{n}.item{i}")

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all writes succeeded
        for n in range(5):
            for i in range(10):
                assert ctx.get_config(f"thread{n}.item{i}") == i


class TestFakeDataSource:
    """Test the FakeDataSource test double."""

    def test_batch_reading(self):
        """Test reading data in batches."""
        items = [{"id": i} for i in range(10)]
        source = FakeDataSource(items)

        batches = list(source.read_batches(batch_size=3))

        assert len(batches) == 4  # 3, 3, 3, 1
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1
        assert batches[0][0]["id"] == 0
        assert batches[-1][0]["id"] == 9

    def test_read_count_tracking(self):
        """Test that reads are counted."""
        source = FakeDataSource([{"id": 1}])

        assert source.read_count == 0
        list(source.read_batches())
        assert source.read_count == 1
        list(source.read_batches())
        assert source.read_count == 2


class TestFactoryFunctions:
    """Test the factory functions."""

    def test_create_failing_provider(self):
        """Test failing provider factory."""
        provider = create_failing_provider("Custom error")

        with pytest.raises(Exception, match="Custom error"):
            provider.complete("test", Models.GPT4)

    def test_create_slow_provider(self):
        """Test slow provider factory."""
        provider = create_slow_provider(20)

        start = time.time()
        provider.complete("test", Models.GPT4)
        elapsed = time.time() - start

        assert elapsed >= 0.02

    def test_create_registry_with_models(self):
        """Test registry factory with standard models."""
        registry = create_registry_with_models()

        # Should have standard models registered
        assert Models.GPT4 in registry.list_models()
        assert Models.CLAUDE3 in registry.list_models()

        # Should be able to get them
        gpt4 = registry.get_model(Models.GPT4)
        claude = registry.get_model(Models.CLAUDE3)

        assert isinstance(gpt4, FakeProvider)
        assert isinstance(claude, FakeProvider)
