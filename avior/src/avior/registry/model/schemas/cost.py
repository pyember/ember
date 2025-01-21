from pydantic import BaseModel, field_validator


class ModelCost(BaseModel):
    """
    Represents the cost details for a given model:
      - input_cost_per_thousand: cost per 1000 tokens in the prompt
      - output_cost_per_thousand: cost per 1000 tokens in the completion
    """

    input_cost_per_thousand: float = 0.0
    output_cost_per_thousand: float = 0.0

    @field_validator("input_cost_per_thousand", "output_cost_per_thousand")
    def non_negative_cost(cls, v):
        if v < 0:
            raise ValueError("Cost cannot be negative.")
        return v


class RateLimit(BaseModel):
    """
    Represents rate limiting data for a given model:
      - tokens_per_minute
      - requests_per_minute
    """

    tokens_per_minute: int = 0
    requests_per_minute: int = 0

    @field_validator("tokens_per_minute", "requests_per_minute")
    def non_negative_values(cls, v):
        if v < 0:
            raise ValueError("RateLimit cannot be negative.")
        return v
