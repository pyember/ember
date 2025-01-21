from typing import Any, Dict, Optional
from pydantic import BaseModel, field_validator, Field, ValidationInfo

from src.avior.registry.model.schemas.cost import ModelCost, RateLimit
from src.avior.registry.model.schemas.provider_info import ProviderInfo


class ModelInfo(BaseModel):
    """Describes the metadata and configuration required to instantiate a model."""

    model_id: str = Field(...)
    model_name: str = Field(...)
    cost: ModelCost
    rate_limit: RateLimit
    provider: ProviderInfo
    api_key: Optional[str] = None

    @field_validator("api_key", mode="before")
    def must_have_api_key(
        cls, api_key_value: Optional[str], info: ValidationInfo
    ) -> str:
        """Ensures we have some API key (either model-specific or default)."""
        provider_obj = info.data.get("provider")
        if not api_key_value and (not provider_obj or not provider_obj.default_api_key):
            raise ValueError("No API key provided or defaulted.")
        return api_key_value or provider_obj.default_api_key

    def get_api_key(self) -> str:
        """Returns the validated API key."""
        return self.api_key

    def get_base_url(self) -> Optional[str]:
        """Returns the base URL from the provider, if available."""
        return self.provider.base_url

    model_config = {
        "protected_namespaces": (),  # Disable Pydantic's protected namespace checks
    }
