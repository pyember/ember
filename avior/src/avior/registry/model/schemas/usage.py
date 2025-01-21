from pydantic import BaseModel
from typing import Optional


class UsageStats(BaseModel):
    """
    Standard usage stats returned by each provider after a call.
    cost_usd is optional if you want to track cost in USD.
    """

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "UsageStats") -> "UsageStats":
        return UsageStats(
            total_tokens=self.total_tokens + other.total_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


class UsageRecord(BaseModel):
    """Encapsulates usage for a single request."""

    usage_stats: UsageStats


class UsageSummary(BaseModel):
    """Maintains cumulative usage for a model, including total cost if applicable."""

    model_config = {
        "protected_namespaces": (),
    }

    model_name: str
    total_usage: UsageStats = UsageStats()

    @property
    def total_tokens_used(self) -> int:
        return self.total_usage.total_tokens

    def add_record(self, record: UsageRecord) -> None:
        self.total_usage = self.total_usage + record.usage_stats
