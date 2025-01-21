from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from src.avior.registry.model.schemas.usage import UsageStats


class ChatRequest(BaseModel):
    """
    Universal ChatRequest for Avior. High-level calls can use this,
    and each provider will adapt it into their own param class.
    """
    prompt: str
    context: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    # Arbitrary provider-specific params can be passed here
    provider_params: Dict[str, Any] = {}


class ChatResponse(BaseModel):
    """
    Standard response wrapper, capturing the model output,
    raw provider output, and usage statistics.
    """
    data: str
    raw_output: Any
    usage: Optional["UsageStats"] = None


class BaseChatParameters(BaseModel):
    """
    Base param class that providers can inherit to handle
    commonly needed fields: prompt, context, temperature, etc.
    """
    prompt: str
    context: Optional[str] = None
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2.0)
    max_tokens: Optional[int] = None

    def build_prompt(self) -> str:
        """
        Optionally unify system + user prompt here.
        Subclasses can override if needed for special formatting.
        """
        if self.context:
            return f"{self.context}\n\n{self.prompt}"
        return self.prompt