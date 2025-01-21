from pydantic import BaseModel
from typing import Optional, Dict


class ProviderInfo(BaseModel):
    """
    Encapsulates metadata about a provider (OpenAI, Anthropic, etc.):
      - name: The provider name, e.g. "OpenAI"
      - default_api_key: A fallback API key to use
      - base_url: Optional custom endpoint
      - custom_args: Dictionary for any extra provider-specific configs
    """

    name: str
    default_api_key: Optional[str] = None
    base_url: Optional[str] = None
    custom_args: Dict[str, str] = {}
