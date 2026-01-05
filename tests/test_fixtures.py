"""Test our fixtures to ensure they work correctly.

Even test infrastructure needs tests!
"""

import json

import pytest
import yaml

from tests.test_constants import APIKeys, Models, TestData
from tests.test_doubles import FakeProvider


class TestProviderFixtures:
    """Test provider-related fixtures."""

    def test_fake_provider_fixture(self, fake_provider):
        """Test basic fake provider fixture."""
        assert isinstance(fake_provider, FakeProvider)

        # Should have default responses
        response = fake_provider.complete(TestData.SIMPLE_PROMPT, Models.GPT4)
        assert response.data == TestData.SIMPLE_RESPONSE

    def test_fake_provider_factory(self, fake_provider_factory):
        """Test provider factory fixture."""
        # Create custom provider
        provider = fake_provider_factory(responses={"custom": "custom response"}, latency_ms=10)

        response = provider.complete("custom", Models.GPT4)
        assert response.data == "custom response"

        # Test latency
        import time

        start = time.time()
        provider.complete("test", Models.GPT4)
        elapsed = time.time() - start
        assert elapsed >= 0.01


class TestRegistryFixtures:
    """Test registry-related fixtures."""

    def test_fake_registry_fixture(self, fake_registry):
        """Test pre-configured registry."""
        # Should have standard models
        assert Models.GPT4 in fake_registry.list_models()
        assert Models.CLAUDE3 in fake_registry.list_models()

        # Should be able to get models
        model = fake_registry.get_model(Models.GPT4)
        assert isinstance(model, FakeProvider)

    def test_empty_registry_fixture(self, empty_registry):
        """Test empty registry fixture."""
        assert len(empty_registry.list_models()) == 0

        # Should be able to add providers and models
        empty_registry.providers["test"] = FakeProvider()
        empty_registry.register_model("test-model", "test")

        assert "test-model" in empty_registry.list_models()


class TestContextFixtures:
    """Test context-related fixtures."""

    def test_isolated_context(self, isolated_context):
        """Test isolated context fixture."""
        # Should be clean
        assert isolated_context.get_all_config() == {}

        # Should be able to set/get config
        isolated_context.set_config("test", "value")
        assert isolated_context.get_config("test") == "value"

    def test_context_with_config(self, context_with_config):
        """Test pre-configured context."""
        # Should have default config
        assert context_with_config.get_config("model") == Models.GPT4
        assert context_with_config.get_config("provider") == "openai"
        assert context_with_config.get_config("api_keys.openai") == APIKeys.OPENAI


class TestResponseFixtures:
    """Test response-related fixtures."""

    def test_api_response_factory(self, api_response_factory):
        """Test response factory fixture."""
        # Create default response
        response = api_response_factory.create()
        assert response.data == "Test response"
        assert response.model_id == Models.GPT4

        # Create custom response
        custom = api_response_factory.create(
            content="Custom", model=Models.CLAUDE3, prompt_tokens=50
        )
        assert custom.data == "Custom"
        assert custom.model_id == Models.CLAUDE3
        assert custom.usage.prompt_tokens == 50

        # Create error response
        error = api_response_factory.create_error_response("Network error")
        assert "Network error" in error.data
        assert error.usage.total_tokens == 0


class TestDataFixtures:
    """Test data-related fixtures."""

    def test_sample_data(self, sample_data):
        """Test sample data fixture."""
        assert len(sample_data) == 3
        assert sample_data[0]["id"] == 1
        assert sample_data[0]["text"] == "First item"

    def test_fake_data_source(self, fake_data_source):
        """Test fake data source fixture."""
        batches = list(fake_data_source.read_batches(batch_size=2))
        assert len(batches) == 2  # 3 items with batch size 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1


class TestFileFixtures:
    """Test file-related fixtures."""

    def test_temp_json_file(self, temp_json_file):
        """Test temporary JSON file fixture."""
        assert temp_json_file.exists()

        with open(temp_json_file) as f:
            data = json.load(f)

        assert len(data) == 3
        assert data[0]["text"] == "First item"

    def test_temp_yaml_file(self, temp_yaml_file):
        """Test temporary YAML file fixture."""
        assert temp_yaml_file.exists()

        with open(temp_yaml_file) as f:
            data = yaml.safe_load(f)

        assert data["model"] == Models.GPT4
        assert data["config"]["temperature"] == 0.7

    def test_temp_csv_file(self, temp_csv_file):
        """Test temporary CSV file fixture."""
        assert temp_csv_file.exists()

        content = temp_csv_file.read_text()
        assert "text,label" in content
        assert "First item,A" in content


class TestEnvironmentFixtures:
    """Test environment-related fixtures."""

    def test_clean_env(self, clean_env):
        """Test clean environment fixture."""
        import os

        # Should have no API keys
        assert os.environ.get(APIKeys.ENV_OPENAI) is None
        assert os.environ.get(APIKeys.ENV_ANTHROPIC) is None
        assert os.environ.get(APIKeys.ENV_GOOGLE) is None

    def test_test_api_keys(self, test_api_keys):
        """Test API keys fixture."""
        import os

        # Should have test API keys
        assert os.environ.get(APIKeys.ENV_OPENAI) == APIKeys.OPENAI
        assert os.environ.get(APIKeys.ENV_ANTHROPIC) == APIKeys.ANTHROPIC
        assert os.environ.get(APIKeys.ENV_GOOGLE) == APIKeys.GOOGLE


class TestHelperFixtures:
    """Test helper fixtures."""

    def test_benchmark_timer(self, benchmark_timer):
        """Test benchmark timer fixture."""
        import time

        with benchmark_timer as timer:
            time.sleep(0.01)

        # Should measure time correctly
        assert timer.elapsed >= 0.01
        timer.assert_slower_than(0.005)
        timer.assert_faster_than(0.05)

    def test_assert_error_matches(self, assert_error_matches):
        """Test error assertion helper."""
        from tests.test_constants import ErrorPatterns

        # Test with pattern object
        with pytest.raises(ValueError, match="API key") as exc_info:
            raise ValueError("No API key found")

        assert_error_matches(exc_info, ErrorPatterns.MISSING_API_KEY)

        # Test with pattern name
        with pytest.raises(ValueError, match="Model") as exc_info:
            raise ValueError("Model not found")

        assert_error_matches(exc_info, "INVALID_MODEL")

    def test_assert_response_valid(self, assert_response_valid, api_response_factory):
        """Test response validation helper."""
        # Valid response should pass
        response = api_response_factory.create()
        assert_response_valid(response)

        # Also works with error responses
        error_response = api_response_factory.create_error_response()
        assert_response_valid(error_response)

    def test_batch_processor(self, batch_processor):
        """Test batch processor fixture."""
        items = ["hello", "world", "test", "batch"]
        results = batch_processor(items, batch_size=2)

        assert results == ["HELLO", "WORLD", "TEST", "BATCH"]
        assert len(results) == len(items)
