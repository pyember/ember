"""Tests for ModelRegistry."""

import importlib
import queue
import threading
from typing import Any, Optional
from unittest.mock import patch

import pytest

from ember._internal.exceptions import ModelNotFoundError, ModelProviderError
from ember.models import ModelRegistry
from ember.models.providers.base import BaseProvider
from ember.models.schemas import ChatResponse, UsageStats

# Import our test infrastructure
from tests.test_constants import ErrorPatterns, Models
from tests.test_doubles import FakeProvider

pytestmark = pytest.mark.usefixtures("tmp_ctx")


class TestModelRegistry:
    """Test the ModelRegistry class behavior."""

    def test_initialization(self):
        """Test registry starts with no models."""
        registry = ModelRegistry()

        # Should start empty (test behavior, not _models attribute)
        assert registry.list_models() == []

    @pytest.mark.parametrize(
        "model_id",
        [
            pytest.param(Models.GPT4, id="gpt4"),
            pytest.param(Models.CLAUDE3, id="claude3"),
            pytest.param(Models.GEMINI_PRO, id="gemini"),
        ],
    )
    def test_get_model_creates_and_caches(self, model_id):
        """Test lazy model instantiation and caching behavior."""
        from ember._internal.context.runtime import EmberContext

        registry = ModelRegistry(context=EmberContext.current())

        # Mock the provider resolution
        with patch("ember.models.runtime.registry.resolve_model_id") as mock_resolve:
            with patch("ember.models.runtime.registry.get_provider_class") as mock_get_class:
                # Setup mocks to return a fake provider
                if "gpt" in model_id:
                    provider_name = "openai"
                elif "claude" in model_id:
                    provider_name = "anthropic"
                else:
                    provider_name = "google"

                mock_resolve.return_value = (provider_name, model_id)
                mock_get_class.return_value = FakeProvider

                # First access
                EmberContext.current().set_config(f"providers.{provider_name}.api_key", "test-key")
                model1 = registry.get_model(model_id)

                # Should return a model
                assert model1 is not None

                # Second access should return same instance (caching behavior)
                model2 = registry.get_model(model_id)
                assert model1 is model2

    def test_thread_safety_behavior(self):
        """Test concurrent access returns consistent results."""
        from ember._internal.context.runtime import EmberContext

        ctx = EmberContext.current()
        ctx.set_config("providers.openai.api_key", "test-key")
        registry = ModelRegistry(context=ctx)
        results_queue = queue.Queue()
        errors_queue = queue.Queue()

        def get_model_thread():
            try:
                model = registry.get_model(Models.GPT4)
                results_queue.put(id(model))  # Store object ID to check same instance
            except Exception as e:
                errors_queue.put(e)

        with patch(
            "ember.models.runtime.registry.resolve_model_id",
            return_value=("openai", Models.GPT4),
        ):
            with patch(
                "ember.models.runtime.registry.get_provider_class",
                return_value=FakeProvider,
            ):
                # Create multiple threads
                threads = [threading.Thread(target=get_model_thread) for _ in range(10)]

                # Start all threads
                for t in threads:
                    t.start()

                # Wait for completion
                for t in threads:
                    t.join()

        # Check results
        assert errors_queue.empty(), "No errors should occur"

        # All threads should get same model instance
        model_ids = []
        while not results_queue.empty():
            model_ids.append(results_queue.get())

        assert len(model_ids) == 10
        assert all(mid == model_ids[0] for mid in model_ids), "All threads should get same instance"

    def test_clear_cache_behavior(self):
        """Test that clear_cache removes cached models."""
        from ember._internal.context.runtime import EmberContext

        ctx = EmberContext.current()
        ctx.set_config("providers.openai.api_key", "test-key")
        registry = ModelRegistry(context=ctx)

        # Setup environment
        with patch("ember.models.runtime.registry.resolve_model_id") as mock_resolve:
            with patch("ember.models.runtime.registry.get_provider_class") as mock_get_class:
                mock_resolve.return_value = ("openai", Models.GPT4)
                mock_get_class.return_value = FakeProvider

                # Get a model (should cache it)
                model1 = registry.get_model(Models.GPT4)

                # Clear cache
                registry.clear_cache()

                # Get model again - should be different instance
                model2 = registry.get_model(Models.GPT4)

                # Different instances indicate cache was cleared
                assert model1 is not model2

    def test_list_models_behavior(self):
        """Test listing cached models."""
        from ember._internal.context.runtime import EmberContext

        ctx = EmberContext.current()
        ctx.set_config("providers.openai.api_key", "test-key")
        ctx.set_config("providers.anthropic.api_key", "test-key")
        registry = ModelRegistry(context=ctx)

        # Initially empty
        assert registry.list_models() == []

        # Add some models by getting them
        with patch("ember.models.runtime.registry.resolve_model_id") as mock_resolve:
            with patch("ember.models.runtime.registry.get_provider_class") as mock_get_class:
                mock_get_class.return_value = FakeProvider

                # Get different models
                for model_id in [Models.GPT4, Models.CLAUDE3]:
                    provider = "openai" if "gpt" in model_id else "anthropic"
                    mock_resolve.return_value = (provider, model_id)
                    registry.get_model(model_id)

        # Should list the cached models
        models = registry.list_models()
        assert len(models) == 2
        assert Models.GPT4 in models
        assert Models.CLAUDE3 in models

    @pytest.mark.parametrize(
        "error_scenario,expected_error,error_pattern",
        [
            pytest.param(
                "unknown_provider",
                ModelNotFoundError,
                ErrorPatterns.INVALID_MODEL,
                id="unknown-provider",
            ),
            pytest.param(
                "missing_api_key",
                ModelProviderError,
                ErrorPatterns.MISSING_API_KEY,
                id="missing-api-key",
            ),
        ],
    )
    def test_error_scenarios(self, error_scenario, expected_error, error_pattern):
        """Test various error scenarios."""
        from ember._internal.context.runtime import EmberContext

        registry = ModelRegistry(context=EmberContext.current())

        if error_scenario == "unknown_provider":
            with patch("ember.models.runtime.registry.resolve_model_id") as mock_resolve:
                mock_resolve.return_value = ("unknown", "some-model")

                with pytest.raises(expected_error) as exc_info:
                    registry.get_model("some-model")

                assert error_pattern.search(str(exc_info.value))

        elif error_scenario == "missing_api_key":
            # Test that missing API key raises ModelProviderError
            with patch("ember.core.setup_launcher.launch_setup_if_needed", return_value=None):
                with pytest.raises(expected_error) as exc_info:
                    registry.get_model(Models.GPT4)

                assert error_pattern.search(str(exc_info.value))

    def test_canonicalizes_gpt5_reasoning_alias(self, monkeypatch):
        """Aliases like gpt-5-high map to gpt-5 and set reasoning defaults."""

        registry = ModelRegistry()
        calls: list[dict[str, object]] = []

        class RecordingProvider(BaseProvider):
            requires_api_key = False

            def __init__(self, api_key=None):
                super().__init__(api_key)

            def complete(self, prompt: str, model: str, **kwargs: Any) -> ChatResponse:  # type: ignore[override]
                calls.append({"prompt": prompt, "model": model, "kwargs": dict(kwargs)})
                return ChatResponse(data="ok", model_id=model, usage=UsageStats())

            def _get_api_key_from_env(self) -> Optional[str]:  # type: ignore[override]
                return None

        registry_module = importlib.import_module("ember.models.runtime.registry")
        monkeypatch.setattr(
            registry_module,
            "get_provider_class",
            lambda provider_name: RecordingProvider,
        )

        response = registry.invoke_model("openai/gpt-5-high", "first")
        assert response.data == "ok"
        assert calls
        assert calls[0]["model"] == "gpt-5"
        assert calls[0]["kwargs"].get("reasoning") == {"effort": "high"}

        calls.clear()
        response = registry.invoke_model("gpt-5-minimal", "second")
        assert response.data == "ok"
        assert calls[0]["model"] == "gpt-5"
        assert calls[0]["kwargs"].get("reasoning") == {"effort": "minimal"}

        calls.clear()
        response = registry.invoke_model("gpt-5-high", "override", reasoning={"effort": "low"})
        assert response.data == "ok"
        assert calls[0]["model"] == "gpt-5"
        assert calls[0]["kwargs"].get("reasoning") == {"effort": "low"}

    def test_responses_canonicalizes_gpt5_alias(self, monkeypatch):
        """Responses API invocations should rewrite GPT-5 aliases."""

        registry = ModelRegistry()
        calls: list[dict[str, object]] = []

        class RecordingProvider(BaseProvider):
            supports_responses_api = True
            requires_api_key = False

            def __init__(self, api_key=None):
                super().__init__(api_key)

            def complete(self, prompt: str, model: str, **kwargs):  # type: ignore[override]
                return ChatResponse(data="ok", model_id=model, usage=UsageStats())

            def complete_responses_payload(self, payload, **kwargs):  # type: ignore[override]
                calls.append({"payload": dict(payload), "kwargs": dict(kwargs)})
                return ChatResponse(data="ok", model_id=payload.get("model"), usage=UsageStats())

            def _get_api_key_from_env(self):  # type: ignore[override]
                return None

        registry_module = importlib.import_module("ember.models.runtime.registry")
        monkeypatch.setattr(
            registry_module,
            "get_provider_class",
            lambda provider_name: RecordingProvider,
        )

        payload = {
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            ]
        }

        response = registry.invoke_responses("gpt-5-high", payload)

        assert response.data == "ok"
        assert calls
        recorded_payload = calls[0]["payload"]
        assert recorded_payload["model"] == "gpt-5"
        assert recorded_payload["reasoning"]["effort"] == "high"

    @pytest.mark.parametrize(
        "api_key",
        [
            pytest.param("standard-key", id="standard-config"),
        ],
    )
    def test_api_key_resolution(self, api_key):
        """Test API key resolution from context configuration."""
        from ember._internal.context.runtime import EmberContext

        ctx = EmberContext.current()
        ctx.set_config("providers.openai.api_key", api_key)
        registry = ModelRegistry(context=ctx)

        with patch("ember.models.runtime.registry.resolve_model_id") as mock_resolve:
            with patch("ember.models.runtime.registry.get_provider_class") as mock_get_class:
                mock_resolve.return_value = ("openai", Models.GPT4)

                # Mock the provider class to capture the API key
                class CapturingProvider:
                    def __init__(self, api_key=None, **kwargs):
                        self.api_key = api_key

                mock_get_class.return_value = CapturingProvider

        model = registry.get_model(Models.GPT4)

        # Check the provider got the expected key
        assert model.api_key == api_key

    def test_stream_model_yields_chunks_and_final_response(self):
        """stream_model should yield text chunks and return final ChatResponse."""

        registry = ModelRegistry()

        class StreamingProvider(BaseProvider):
            requires_api_key = False

            def __init__(self, api_key=None):
                super().__init__(api_key)

            def complete(self, prompt: str, model: str, **kwargs: Any) -> ChatResponse:  # type: ignore[override]
                raise AssertionError("stream_model should not call complete")

            def stream_complete(self, prompt: str, model: str, **kwargs: Any):  # type: ignore[override]
                def _generator():
                    yield "foo"
                    yield "bar"
                    return ChatResponse(
                        data="foobar",
                        model_id=model,
                        usage=UsageStats(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                    )

                return _generator()

            def _get_api_key_from_env(self) -> Optional[str]:  # type: ignore[override]
                return None

        with patch(
            "ember.models.runtime.registry.resolve_model_id",
            return_value=("openai", "gpt-5"),
        ):
            with patch.dict(
                "ember.models.providers.PROVIDERS",
                {"openai": StreamingProvider},
                clear=False,
            ):
                provider = registry.get_model("gpt-5")
                assert isinstance(provider, StreamingProvider)

                generator = registry.stream_model("gpt-5", "Hello")

                chunks: list[str] = []
                try:
                    while True:
                        chunks.append(next(generator))
                except StopIteration as stop:
                    final = stop.value

        assert chunks == ["foo", "bar"]
        assert isinstance(final, ChatResponse)
        assert final.data == "foobar"
        assert final.model_id == "gpt-5"
        assert final.usage.total_tokens == 2

    @pytest.mark.parametrize(
        "num_threads",
        [
            pytest.param(10, id="10-threads"),
            pytest.param(20, id="20-threads"),
        ],
    )
    def test_concurrent_different_models(self, num_threads):
        """Test concurrent access to different models."""
        from ember._internal.context.runtime import EmberContext

        ctx = EmberContext.current()
        ctx.set_config("providers.openai.api_key", "test-key")
        ctx.set_config("providers.anthropic.api_key", "test-key")
        registry = ModelRegistry(context=ctx)
        results_queue = queue.Queue()
        errors_queue = queue.Queue()

        # Use a fixed set of valid models
        test_models = [Models.GPT4, Models.GPT35, Models.CLAUDE3]

        def get_model_thread(model_idx):
            try:
                model_id = test_models[model_idx % len(test_models)]
                model = registry.get_model(model_id)
                results_queue.put((model_id, id(model)))
            except Exception as e:
                errors_queue.put(e)

        # Create threads accessing different models
        def _resolve(model: str) -> tuple[str, str]:
            provider = "openai" if "gpt" in model.lower() else "anthropic"
            return provider, model

        with patch("ember.models.runtime.registry.resolve_model_id", side_effect=_resolve):
            with patch(
                "ember.models.runtime.registry.get_provider_class",
                return_value=FakeProvider,
            ):
                threads = []
                for i in range(num_threads):
                    t = threading.Thread(target=get_model_thread, args=(i,))
                    threads.append(t)
                    t.start()

                # Wait for completion
                for t in threads:
                    t.join()

        # Check results
        if not errors_queue.empty():
            errors = []
            while not errors_queue.empty():
                errors.append(str(errors_queue.get()))
            pytest.fail(f"Errors occurred: {errors}")

        # Group results by model
        model_instances = {}
        while not results_queue.empty():
            model_id, instance_id = results_queue.get()
            if model_id not in model_instances:
                model_instances[model_id] = set()
            model_instances[model_id].add(instance_id)

        # Each model should have only one instance (proper caching)
        for model_id, instances in model_instances.items():
            assert len(instances) == 1, f"Model {model_id} should have single instance"


class TestModelCatalog:
    def test_catalog_includes_gpt5_codex(self):
        from ember.models.catalog import MODEL_CATALOG

        assert "gpt-5-codex" in MODEL_CATALOG
        info = MODEL_CATALOG["gpt-5-codex"]
        assert info.provider == "openai"

