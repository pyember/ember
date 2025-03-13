from typing import Any, Dict, Optional
from pydantic import Field, field_validator, ValidationInfo, ConfigDict

from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.base.schemas.provider_info import ProviderInfo
from ember.core.types.ember_model import EmberModel


class ModelInfo(EmberModel):
    """
    Metadata and configuration for instantiating a model.

    Attributes:
        id (str): Unique identifier for the model.
        name (str): Human-readable name of the model.
        cost (ModelCost): Cost details associated with the model.
        rate_limit (RateLimit): Rate limiting parameters for model usage.
        provider (ProviderInfo): Provider information containing defaults and endpoints.
        api_key (Optional[str]): API key for authentication. If omitted, the provider's default API key is used.
    """

    model_config = ConfigDict(
        protected_namespaces=(),  # Disable Pydantic's protected namespace checks.
    )

    id: str = Field(...)
    name: str = Field(...)
    cost: ModelCost
    rate_limit: RateLimit
    provider: ProviderInfo
    api_key: Optional[str] = None

    @property
    def model_id(self) -> str:
        """Alias for id, using a more descriptive name."""
        return self.id

    @property
    def model_name(self) -> str:
        """Alias for name, using a more descriptive name."""
        return self.name

    @field_validator("api_key", mode="before")
    def validate_api_key(cls, api_key: Optional[str], info: ValidationInfo) -> str:
        """
        Ensures an API key is provided, either explicitly or via the provider.

        This validator checks if an API key is supplied. If not, it attempts to obtain a default
        API key from the associated provider. A ValueError is raised if neither is available.

        Args:
            api_key: The API key provided before validation.
            info: Validation context containing additional field data.

        Returns:
            A valid API key.

        Raises:
            ValueError: If no API key is provided and the provider lacks a default.
        """
        provider_obj = info.data.get("provider")
        if not api_key and (not provider_obj or not provider_obj.default_api_key):
            raise ValueError("No API key provided or defaulted.")
        return api_key or provider_obj.default_api_key

    def get_api_key(self) -> str:
        """
        Retrieves the validated API key.

        Returns:
            The API key to be used for authentication.
        """
        # Assert that the api_key is set following validation.
        assert (
            self.api_key is not None
        ), "The API key must have been set by the validator."
        return self.api_key

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
