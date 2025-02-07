import unittest
from unittest.mock import Mock, patch
import os
from typing import Optional
import google

from ember.registry.operators.operator_base import OperatorContext
from ember.registry.model.model_registry import (
    ModelRegistry,
    OpenAIModel,
    AnthropicModel,
    GeminiModel,
    ProviderInfo,
    ModelInfo,
    ModelCost,
    RateLimit,
    UsageRecord,
    UsageSummary,
    InvalidInputError,
    ModelExecutionError,
)
from google.auth.exceptions import DefaultCredentialsError
from google.api_core.exceptions import PermissionDenied, GoogleAPIError


class TestModelRegistry(unittest.TestCase):
    """Test suite for the ModelRegistry class."""

    def setUp(self) -> None:
        """Setting up the test environment before each test method."""
        self.model_registry = ModelRegistry()
        self.model_id = "test_model"
        self.mock_model = Mock(spec=OpenAIModel)
        self.model_info = ModelInfo(
            model_id=self.model_id,
            model_name="Test Model",
            provider=ProviderInfo(name="Test Provider", default_api_key="test_api_key"),
            cost=ModelCost(input_cost_per_thousand=0.01, output_cost_per_thousand=0.02),
            rate_limit=RateLimit(tokens_per_minute=1000, requests_per_minute=10),
        )

    def test_register_model(self) -> None:
        """Testing the register_model method of ModelRegistry."""
        self.model_registry.register_model(self.model_info, self.mock_model)
        self.assertIn(self.model_id, self.model_registry.models)
        self.assertEqual(self.model_registry.models[self.model_id], self.mock_model)
        self.assertIn(self.model_id, self.model_registry.usage_summaries)
        self.assertIsInstance(
            self.model_registry.usage_summaries[self.model_id], UsageSummary
        )

    def test_get_model(self) -> None:
        """Testing the get_model method of ModelRegistry."""
        self.model_registry.register_model(self.model_info, self.mock_model)
        retrieved_model = self.model_registry.get_model(self.model_id)
        self.assertIsNotNone(
            retrieved_model
        )  # First, checking if the retrieved model is not None
        self.assertEqual(retrieved_model, self.mock_model)

    def test_get_nonexistent_model(self) -> None:
        """Testing the get_model method with a nonexistent model."""
        retrieved_model = self.model_registry.get_model("nonexistent_model")
        self.assertIsNone(retrieved_model)

    def test_list_models(self) -> None:
        """Testing the list_models method of ModelRegistry."""
        self.model_registry.register_model(self.model_info, self.mock_model)
        model_list = self.model_registry.list_models()
        self.assertIsInstance(model_list, list)
        self.assertIn(self.model_id, model_list)

    def test_add_usage_record(self) -> None:
        """Testing the add_usage_record method of ModelRegistry."""
        self.model_registry.register_model(self.model_info, self.mock_model)
        usage_record = UsageRecord(
            model_name=self.model_id,
            request_time=1000.0,
            token_count=100,
            request_latency=0.5,
            cost=0.1,
        )
        self.model_registry.add_usage_record(self.model_id, usage_record)
        summary = self.model_registry.usage_summaries[self.model_id]
        self.assertEqual(summary.total_tokens, 100)
        self.assertEqual(summary.total_requests, 1)
        self.assertEqual(summary.total_cost, 0.1)

    def test_add_usage_record_nonexistent_model(self) -> None:
        """Testing the add_usage_record method with a nonexistent model."""
        usage_record = UsageRecord(
            model_name="nonexistent_model",
            request_time=1000.0,
            token_count=100,
            request_latency=0.5,
            cost=0.1,
        )
        with self.assertRaises(ValueError):
            self.model_registry.add_usage_record("nonexistent_model", usage_record)

    def test_get_usage_summary(self) -> None:
        """Testing the get_usage_summary method of ModelRegistry."""
        self.model_registry.register_model(self.model_info, self.mock_model)
        retrieved_summary = self.model_registry.get_usage_summary(self.model_id)
        self.assertIsInstance(retrieved_summary, UsageSummary)
        self.assertEqual(retrieved_summary.model_name, "Test Model")

    def test_get_usage_summary_nonexistent_model(self) -> None:
        """Testing the get_usage_summary method with a nonexistent model."""
        retrieved_summary = self.model_registry.get_usage_summary("nonexistent_model")
        self.assertIsNone(retrieved_summary)


# TODO: Add additional method tests for OpenAIModel, AnthropicModel, and GeminiModel. Test more of the functionality beyond execute.
class TestOpenAIModel(unittest.TestCase):
    """Test suite for the OpenAIModel class."""

    def setUp(self) -> None:
        """Setting up the test environment before each test method."""
        # Setting up the API key from local environment
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        # Set the API key here
        self.model_info = ModelInfo(
            model_id="gpt-4o",
            model_name="gpt-4o",
            provider=ProviderInfo(name="OpenAI", default_api_key=self.api_key),
            cost=ModelCost(input_cost_per_thousand=0.03, output_cost_per_thousand=0.06),
            rate_limit=RateLimit(tokens_per_minute=1000, requests_per_minute=10),
        )
        self.model = OpenAIModel(self.model_info)

    def test_execute(self) -> None:
        """Testing the execute method of OpenAIModel."""
        input = OperatorContext(
            query="What is the capital of France? Explain and give a brief 5 word history."
        )
        processed_result = self.model.forward(input)

        # Asserting that the result contains relevant information
        self.assertIn(
            "Paris", processed_result, "Expected 'Paris' to be in the response"
        )

        # Checking if the response is a non-empty string
        self.assertTrue(
            isinstance(processed_result, str) and len(processed_result) > 0,
            "Expected a non-empty string response",
        )

        # Verifying that the response contains more than just the word "Paris"
        self.assertTrue(
            len(processed_result.split()) > 1,
            "Expected a more detailed response than just 'Paris'",
        )


class TestAnthropicModel(unittest.TestCase):
    """Test suite for the AnthropicModel class."""

    def setUp(self) -> None:
        """Set up the test environment before each test method."""
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            self.skipTest("ANTHROPIC_API_KEY not found in environment variables")

        self.model_info = ModelInfo(
            model_id="claude-3-5-sonnet-20240620",
            model_name="claude-3-5-sonnet-20240620",
            provider=ProviderInfo(name="Anthropic", default_api_key=self.api_key),
            cost=ModelCost(input_cost_per_thousand=0.01, output_cost_per_thousand=0.03),
            rate_limit=RateLimit(tokens_per_minute=2000, requests_per_minute=20),
        )
        self.model = AnthropicModel(self.model_info)

    def test_execute(self) -> None:
        """Test the execute method of AnthropicModel."""
        input = OperatorContext(
            query="What is the capital of France? Explain and give a brief 5-word history."
        )

        result = self.model.forward(input)

        # Assert that the result contains relevant information
        self.assertIn("Paris", result, "Expected 'Paris' to be in the response")

        # Check if the response is a non-empty string
        self.assertTrue(
            isinstance(result, str) and len(result) > 0,
            "Expected a non-empty string response",
        )

        # Verify that the response contains more than just the word "Paris"
        self.assertTrue(
            len(result.split()) > 1,
            "Expected a more detailed response than just 'Paris'",
        )


class TestGeminiModel(unittest.TestCase):
    """Test suite for the GeminiModel class."""

    def setUp(self) -> None:
        """Sets up the test environment before each test method."""
        try:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                self.skipTest("GOOGLE_API_KEY not found in environment variables")

            self.model_info = ModelInfo(
                model_id="gemini-1.5-pro",
                model_name="gemini-1.5-pro",
                provider=ProviderInfo(name="Google", default_api_key=google_api_key),
                cost=ModelCost(
                    input_cost_per_thousand=0.0004, output_cost_per_thousand=0.0016
                ),
                rate_limit=RateLimit(
                    tokens_per_minute=4000000, requests_per_minute=1000
                ),
            )
            self.model = GeminiModel(self.model_info)
        except ModelExecutionError as e:
            self.skipTest(
                f"Model execution error during GeminiModel initialization: {e}"
            )
        except EnvironmentError as e:
            self.skipTest(str(e))
        except Exception as e:
            self.skipTest(f"Failed to initialize GeminiModel: {e}")

    def test_execute(self) -> None:
        """Tests the execute method of GeminiModel."""
        input = OperatorContext(
            query="What is the largest planet in our solar system? Explain briefly."
        )

        result = self.model.forward(
            input,
            temperature=0.7,
            max_output_tokens=256,
        )

        self.assertIsInstance(result, str, "Expected the result to be a string.")
        self.assertTrue(result, "Expected a non-empty response.")
        self.assertIn("Jupiter", result, "Expected 'Jupiter' to be in the response.")
        self.assertGreater(len(result.split()), 5, "Expected a detailed explanation.")


if __name__ == "__main__":
    unittest.main()
