from pydantic import BaseModel, FieldValidationInfo, field_validator


class ModelCost(BaseModel):
    """Represents the cost details for a given model.

    Attributes:
        input_cost_per_thousand (float): Cost per 1000 tokens in the prompt.
        output_cost_per_thousand (float): Cost per 1000 tokens in the completion.
    """

    input_cost_per_thousand: float = 0.0
    output_cost_per_thousand: float = 0.0

    @field_validator(
        fields=("input_cost_per_thousand", "output_cost_per_thousand"), mode="after"
    )
    def validate_non_negative_cost(
        cls, value: float, info: FieldValidationInfo
    ) -> float:
        """Validates that a cost value is not negative.

        Args:
            value (float): The cost value to validate.
            info (FieldValidationInfo): Additional context for the field being validated.

        Raises:
            ValueError: If the cost value is negative.

        Returns:
            float: The validated non-negative cost value.
        """
        if value < 0:
            raise ValueError(
                f"{info.field_name} must be non-negative; received {value}."
            )
        return value


class RateLimit(BaseModel):
    """Represents rate-limiting data for a given model.

    Attributes:
        tokens_per_minute (int): Maximum tokens allowed per minute.
        requests_per_minute (int): Maximum requests allowed per minute.
    """

    tokens_per_minute: int = 0
    requests_per_minute: int = 0

    @field_validator(fields=("tokens_per_minute", "requests_per_minute"), mode="after")
    def validate_non_negative_rate(cls, value: int, info: FieldValidationInfo) -> int:
        """Validates that a rate limit value is not negative.

        Args:
            value (int): The rate limit value to validate.
            info (FieldValidationInfo): Additional context for the field being validated.

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
