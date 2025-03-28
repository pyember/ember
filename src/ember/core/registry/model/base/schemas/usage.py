from pydantic import BaseModel, ConfigDict


class UsageStats(BaseModel):
    """Contains standard usage statistics returned by a provider.

    Attributes:
        total_tokens (int): Total number of tokens.
        prompt_tokens (int): Number of tokens in the prompt.
        completion_tokens (int): Number of tokens in the completion.
        cost_usd (float): Cost in USD, if applicable.
    """

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "UsageStats") -> "UsageStats":
        """Aggregates this usage statistics instance with another.

        Args:
            other (UsageStats): Another usage statistics instance to combine.

        Returns:
            UsageStats: A new instance with aggregated values.
        """
        return UsageStats(
            total_tokens=self.total_tokens + other.total_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    def __add__(self, other: "UsageStats") -> "UsageStats":
        """Overloads the addition operator to combine usage statistics.

        Args:
            other (UsageStats): Another usage statistics instance to combine.

        Returns:
            UsageStats: A new instance with summed usage values.
        """
        return self.add(other)


class UsageRecord(BaseModel):
    """Encapsulates usage data for a single request.

    Attributes:
        usage_stats (UsageStats): The usage statistics associated with the request.
    """

    usage_stats: UsageStats


class UsageSummary(BaseModel):
    """Maintains cumulative usage for a model, including total cost if applicable.

    Attributes:
        model_name (str): Name of the model.
        total_usage (UsageStats): Aggregated usage statistics.
    """

    model_name: str
    total_usage: UsageStats = UsageStats()
    model_config = ConfigDict(protected_namespaces=())

    @property
    def total_tokens_used(self) -> int:
        """Retrieves the total number of tokens used across all aggregated usage.

        Returns:
            int: Cumulative total token count.
        """
        return self.total_usage.total_tokens

    def add_usage_record(self, record: UsageRecord) -> None:
        """Aggregates a usage record into the cumulative usage summary.

        Args:
            record (UsageRecord): A single usage record to incorporate.
        """
        self.total_usage = self.total_usage.add(record.usage_stats)
