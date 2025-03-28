from typing import Optional

from pydantic import ConfigDict, Field, ValidationInfo, field_validator

from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.types.ember_model import EmberModel


class ModelInfo(EmberModel):
    """
    Metadata and configuration for instantiating a model.

    Attributes:
        id (str): Unique identifier for the model.
        name (Optional[str]): Human-readable name of the model. If omitted, defaults to the value of id.
        cost (ModelCost, optional): Cost details associated with the model. Defaults to zero cost.
        rate_limit (RateLimit, optional): Rate limiting parameters for model usage. Defaults to no rate limits.
        provider (ProviderInfo): Provider information containing defaults and endpoints.
        api_key (Optional[str]): API key for authentication. If omitted, the provider's default API key is used.
        context_window (Optional[int]): Maximum context window size in tokens.
    """

    model_config = ConfigDict(
        protected_namespaces=(),  # Disable Pydantic's protected namespace checks.
    )

    id: str = Field(...)
    name: Optional[str] = None
    cost: ModelCost = Field(default_factory=ModelCost)
    rate_limit: RateLimit = Field(default_factory=RateLimit)
    provider: ProviderInfo
    api_key: Optional[str] = None
    context_window: Optional[int] = None

    @property
    def model_id(self) -> str:
        """Alias for id, using a more descriptive name."""
        return self.id

    @property
    def model_name(self) -> str:
        """Alias for name, using a more descriptive name."""
        return self.name

    @field_validator("name", mode="before")
    def default_name_to_id(cls, name: Optional[str], info: ValidationInfo) -> str:
        """
        Sets the name to the value of id when name is not provided.

        Args:
            name: The name value provided before validation.
            info: Validation context containing additional field data.

        Returns:
            The name to use, defaulting to id if none is provided.
        """
        if not name and "id" in info.data:
            return info.data["id"]
        return name or ""

    @field_validator("api_key", mode="after")
    def validate_api_key(
        cls, api_key: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        """
        Ensures an API key is available, either explicitly or via the provider.

        This validator checks if an API key is supplied. If not, it attempts to obtain a default
        API key from the associated provider. A ValueError is raised if neither is available.

        Args:
            api_key: The API key provided before validation.
            info: Validation context containing additional field data.

        Returns:
            A valid API key or None if we'll use the provider's default.

        Raises:
            ValueError: If no API key is provided and the provider lacks a default.
        """
        provider_obj = info.data.get("provider")

        # If we already have an API key, just return it
        if api_key:
            return api_key

        # Otherwise, check if provider has a default key
        if provider_obj and provider_obj.default_api_key:
            # Return the provider's default key
            return provider_obj.default_api_key

        # Check for environment variables later, but for now raise an error if both are None
        if provider_obj and provider_obj.default_api_key is None and api_key is None:
            model_id = info.data.get("id", "unknown")
            raise ValueError(
                f"No API key available for model {model_id}. "
                f"Please set via API, provider default, or environment variable."
            )

        # Return None if we'll check environment variables later
        return None

    def get_api_key(self) -> str:
        """
        Retrieves the API key, first checking the instance then fallback to provider's default.

        Returns:
            The API key to be used for authentication.

        Raises:
            ValueError: If no API key is available from any source.
        """
        # First check if api_key is set directly on this instance
        if self.api_key is not None:
            return self.api_key

        # Then check if we can get it from provider's default
        if self.provider and self.provider.default_api_key:
            return self.provider.default_api_key

        # Last resort - check environment variables based on provider name
        import os

        provider_name = self.provider.name.upper() if self.provider else ""
        env_var_name = f"{provider_name}_API_KEY"
        env_api_key = os.environ.get(env_var_name)

        if env_api_key:
            # Cache it for future calls
            self.api_key = env_api_key
            return env_api_key

        # If we got here, no API key is available
        raise ValueError(
            f"No API key available for model {self.id}. "
            f"Please set via API, provider default, or {env_var_name} environment variable."
        )

    def get_base_url(self) -> Optional[str]:
        """
        Retrieves the base URL from the provider, if it exists.

        Returns:
            The base URL specified by the provider, or None if not available.
        """
        return self.provider.base_url

    def __str__(self) -> str:
        # Avoid exposing API keys
        return (
            f"ModelInfo(id={self.id}, name={self.name}, provider={self.provider.name})"
        )

    def __repr__(self) -> str:
        # Reuse the safe string representation
        return self.__str__()
