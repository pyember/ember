from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import time


class ModelCost(BaseModel):
    """Represents the cost structure for a model."""

    input_cost_per_thousand: float = 0.0
    output_cost_per_thousand: float = 0.0


class RateLimit(BaseModel):
    """Represents rate-limiting constraints for a model."""

    tokens_per_minute: int = 0
    requests_per_minute: int = 0


class UsageRecord(BaseModel):
    """Represents a single usage record for a model request."""

    tokens_used: int
    request_id: str = Field(default_factory=lambda: f"req-{int(time.time() * 1000)}")


class UsageSummary(BaseModel):
    """Maintains a list of usage records and associated metadata."""

    model_name: str
    window_duration: int  # e.g., 60 seconds
    records: List[UsageRecord] = Field(default_factory=list)

    def add_record(self, record: UsageRecord) -> None:
        self.records.append(record)

    def total_tokens_used(self) -> int:
        """Sums all tokens used in the window for quick reporting."""
        return sum(rec.tokens_used for rec in self.records)


class ProviderInfo(BaseModel):
    """Basic provider information and configuration details."""

    name: str
    default_api_key: Optional[str] = None
    base_url: Optional[str] = None
    custom_args: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    """Aggregates all metadata to interact with a specific model instance."""

    model_id: str
    model_name: str
    cost: ModelCost
    rate_limit: RateLimit
    provider: ProviderInfo
    api_key: Optional[str] = None

    def get_api_key(self) -> str:
        return self.api_key or (self.provider.default_api_key or "")

    def get_base_url(self) -> Optional[str]:
        return self.provider.base_url

    def get_custom_args(self) -> Dict[str, Any]:
        return self.provider.custom_args or {}
