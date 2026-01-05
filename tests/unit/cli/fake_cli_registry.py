"""Test-specific model registry for CLI tests."""

from typing import Any, List, Optional
from unittest.mock import Mock

from ember.models.schemas import ChatResponse, UsageStats


class FakeCLIModelRegistry:
    """Simplified model registry for testing.

    Provides predictable behavior without external dependencies.
    All models return deterministic responses for testing.
    """

    def __init__(self):
        self._models = {}
        self._usage_records = {}

    def get_model(self, model_id: str) -> Mock:
        """Always returns a mock model - never fails."""
        if model_id not in self._models:
            # Create a mock that behaves like a provider
            mock_model = Mock()
            mock_model.complete = Mock(
                return_value=ChatResponse(
                    data=f"Test response from {model_id}",
                    model_id=model_id,
                    usage=UsageStats(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.001
                    ),
                )
            )
            self._models[model_id] = mock_model
        return self._models[model_id]

    def invoke_model(self, model_id: str, prompt: str, **kwargs: Any) -> ChatResponse:
        """Direct invocation for testing."""
        model = self.get_model(model_id)
        response = model.complete(prompt, model_id, **kwargs)

        # Track usage
        if model_id not in self._usage_records:
            self._usage_records[model_id] = []
        self._usage_records[model_id].append(response.usage)

        return response

    def set_model_response(self, model_id: str, response_data: str):
        """Configure specific response for a model."""
        mock_model = Mock()
        mock_model.complete = Mock(
            return_value=ChatResponse(
                data=response_data,
                model_id=model_id,
                usage=UsageStats(
                    prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.001
                ),
            )
        )
        self._models[model_id] = mock_model

    def set_model_error(self, model_id: str, error: Exception):
        """Configure a model to raise an error."""
        mock_model = Mock()
        mock_model.complete = Mock(side_effect=error)
        self._models[model_id] = mock_model

    def list_models(self) -> List[str]:
        """List all configured models."""
        return list(self._models.keys())

    def clear_cache(self):
        """Clear all models."""
        self._models.clear()
        self._usage_records.clear()

    def get_usage_summary(self, model_id: str) -> Optional[UsageStats]:
        """Get usage summary for testing."""
        records = self._usage_records.get(model_id, [])
        if not records:
            return None

        total = UsageStats()
        for record in records:
            total.add(record)
        return total
