from abc import ABC, abstractmethod
from typing import Any, Optional

from ember.core.registry.model.base.schemas.usage import UsageStats
from ember.core.registry.model.base.schemas.model_info import ModelInfo


class UsageCalculator(ABC):
    """Abstract base class for calculating usage statistics from provider responses.

    This interface defines a contract for computing usage statistics based on raw
    provider output and associated model configuration.
    """

    @abstractmethod
    def calculate(self, *, raw_output: Any, model_info: ModelInfo) -> UsageStats:
        """Compute usage statistics given raw provider output and model configuration.

        Args:
            raw_output (Any): The raw output from the provider, which may include usage data.
            model_info (ModelInfo): The model configuration details, including cost metrics.

        Returns:
            UsageStats: The computed usage statistics.
        """
        raise NotImplementedError


class DefaultUsageCalculator(UsageCalculator):
    """Default implementation for calculating usage statistics assuming token count metadata is provided.

    This implementation extracts token counts using primary and fallback attribute names
    from the raw output, computes associated costs, and returns structured usage statistics.
    """

    @staticmethod
    def _get_token_count(*, usage: Any, primary_field: str, fallback_field: str) -> int:
        """Retrieve the token count from usage using a primary attribute and a fallback.

        Args:
            usage (Any): The usage object containing token count data.
            primary_field (str): The primary attribute name for the token count.
            fallback_field (str): The fallback attribute name if the primary is not present.

        Returns:
            int: The token count extracted from the usage object, or 0 if not found.
        """
        primary_value: Optional[int] = getattr(usage, primary_field, None)
        if primary_value is not None:
            return primary_value

        fallback_value: Optional[int] = getattr(usage, fallback_field, None)
        return fallback_value if fallback_value is not None else 0

    def calculate(self, *, raw_output: Any, model_info: ModelInfo) -> UsageStats:
        """Calculate usage statistics from raw provider output.

        This method extracts token counts using both primary and fallback attribute names,
        computes the input and output costs based on the model configuration, and returns
        an aggregated UsageStats object.

        Args:
            raw_output (Any): The raw output data from the provider, expected to contain usage metadata.
            model_info (ModelInfo): Model configuration including cost metrics.

        Returns:
            UsageStats: The computed usage statistics including token counts and total cost in USD.
        """
        usage: Any = getattr(raw_output, "usage", None) or getattr(
            raw_output, "usage_metadata", None
        )
        if usage is None:
            return UsageStats()

        total_tokens: int = self._get_token_count(
            usage=usage,
            primary_field="total_tokens",
            fallback_field="total_token_count",
        )
        prompt_tokens: int = self._get_token_count(
            usage=usage,
            primary_field="prompt_tokens",
            fallback_field="prompt_token_count",
        )
        completion_tokens: int = self._get_token_count(
            usage=usage,
            primary_field="completion_tokens",
            fallback_field="candidates_token_count",
        )

        input_cost: float = (
            prompt_tokens / 1000.0
        ) * model_info.cost.input_cost_per_thousand
        output_cost: float = (
            completion_tokens / 1000.0
        ) * model_info.cost.output_cost_per_thousand
        total_cost: float = round(input_cost + output_cost, 6)

        return UsageStats(
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=total_cost,
        )
