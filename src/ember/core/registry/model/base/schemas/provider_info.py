from typing import Dict, Optional

from pydantic import BaseModel, Field


class ProviderInfo(BaseModel):
    """Encapsulates metadata about a service provider.

    Attributes:
        name (str): The provider's name (e.g., "OpenAI").
        default_api_key (Optional[str]): Fallback API key if none is provided.
        base_url (Optional[str]): Optional custom URL endpoint.
        custom_args (Dict[str, str]): Additional provider-specific configuration parameters.
    """

    name: str
    default_api_key: Optional[str] = None
    base_url: Optional[str] = None
    custom_args: Dict[str, str] = Field(default_factory=dict)
