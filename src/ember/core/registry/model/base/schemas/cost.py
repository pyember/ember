from pydantic import BaseModel, ValidationInfo, field_validator, model_validator


class ModelCost(BaseModel):
    """Represents the cost details for a given model.

    Attributes:
        input_cost_per_thousand (float): Cost per 1000 tokens in the prompt.
        output_cost_per_thousand (float): Cost per 1000 tokens in the completion.
    """

    input_cost_per_thousand: float = 0.0
    output_cost_per_thousand: float = 0.0

    @property
    def input_cost(self) -> float:
        """Normalized input cost per thousand tokens."""
        # Example: if cost is set to a large number, treat it as cost per million or something
        return (
            self.input_cost_per_thousand
            if self.input_cost_per_thousand < 100
            else self.input_cost_per_thousand / 1000.0
        )

    @property
    def output_cost(self) -> float:
        """Normalized output cost per thousand tokens."""
        return (
            self.output_cost_per_thousand
            if self.output_cost_per_thousand < 100
            else self.output_cost_per_thousand / 1000.0
        )

    @model_validator(mode="after")
    def validate_costs(self):
        """Validate that costs are non-negative."""
        if self.input_cost_per_thousand < 0 or self.output_cost_per_thousand < 0:
            raise ValueError("Costs must be non-negative.")
        return self


class RateLimit(BaseModel):
    """Represents rate-limiting data for a given model.

    Attributes:
        tokens_per_minute (int): Maximum tokens allowed per minute.
        requests_per_minute (int): Maximum requests allowed per minute.
    """

    tokens_per_minute: int = 0
    requests_per_minute: int = 0

    @field_validator("tokens_per_minute", "requests_per_minute", mode="after")
    def validate_non_negative_rate(cls, value: int, info: ValidationInfo) -> int:
        """Validates that a rate limit value is not negative.

        Args:
            value (int): The rate limit value to validate.
            info (ValidationInfo): Additional context for the field being validated.

        Raises:
            ValueError: If the rate limit value is negative.

        Returns:
            int: The validated non-negative rate limit value.
        """
        if value < 0:
            raise ValueError(
                f"{info.field_name} must be non-negative; received {value}."
            )
        return value
