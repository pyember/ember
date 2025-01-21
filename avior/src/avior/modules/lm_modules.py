import logging
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field

from src.avior.registry.model.services.model_service import ModelService
from src.avior.registry.model.services.usage_service import UsageService
from src.avior.registry.model.registry.model_registry import GLOBAL_MODEL_REGISTRY

logger = logging.getLogger(__name__)

class LMModuleConfig(BaseModel):
    """
    Configuration settings for the Language Model module.
    """
    model_id: str = Field(
        "openai:gpt-4o",
        description="String or enum-based identifier for picking the underlying model provider."
    )
    temperature: float = Field(
        1.0, ge=0.0, le=5.0, 
        description="Sampling temperature for model generation."
    )
    max_tokens: Optional[int] = Field(
        None, 
        description="Maximum tokens to generate in a single forward call."
    )
    cot_prompt: Optional[str] = Field(
        None, 
        description="Optional chain-of-thought prompt or format appended to the user's prompt."
    )
    persona: Optional[str] = Field(
        None, 
        description="Optional persona or role context to be prepended to user query."
    )
    # You can add more fields as needed (top_k, top_p, etc.).

def get_default_model_service() -> ModelService:
    """
    Creates and returns a default ModelService if none is provided
    to the LMModule. This uses a new ModelRegistry and UsageService.
    If you need to pre-register default models, do so here.
    """
    registry = GLOBAL_MODEL_REGISTRY
    usage_service = UsageService()
    # Optionally, register default models here:
    # e.g. registry.register_model(<some ModelInfo>)
    return ModelService(registry, usage_service)

class LMModule:
    """
    A highly extensible Language Model module that integrates with a ModelService
    and (optionally) a UsageService to handle usage logging, cost tracking, etc.

    The user typically just calls this module with a string prompt:
        lm_module = LMModule(config, model_service, usage_se rvice)
        response_text = lm_module("Hello, world!")
    """

    def __init__(
        self,
        config: LMModuleConfig,
        model_service: Optional[ModelService] = None,
    ):
        """
        :param config: The LMModuleConfig instance specifying model_name, temperature, etc.
        :param model_service: The ModelService that handles underlying model invocation.
                             If None, we fall back to get_default_model_service().
        """
        if model_service is None:
            model_service = get_default_model_service()

        self.config = config
        self.model_service = model_service
        self._logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """
        Allows direct invocation: `lm_module("Prompt")`.
        Equivalent to `forward(prompt, **kwargs)`.
        """
        return self.forward(prompt, **kwargs)

    def forward(self, prompt: str, **kwargs: Any) -> str:
        """
        Main entry point for generating text from a prompt, delegating
        usage tracking to the ModelService.
        """
        # 1) Merge persona or chain-of-thought into single final prompt
        final_prompt = self._assemble_full_prompt(prompt)

        # 2) Invoke the model, passing just the final prompt
        response = self.model_service.invoke_model(
            self.config.model_id,
            final_prompt,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            # We no longer pass context here
            **kwargs
        )

        # 3) Return final text
        return response.data if hasattr(response, "data") else str(response)

    def _assemble_full_prompt(self, user_prompt: str) -> str:
        """
        Internal helper to merge persona, chain-of-thought, etc.
        into the final prompt text.
        """
        segments = []
        # Prepend persona if present
        if self.config.persona:
            segments.append(f"[Persona: {self.config.persona}]\n")

        # Add user prompt
        segments.append(user_prompt.strip())

        # Append chain-of-thought prompt if present
        if self.config.cot_prompt:
            segments.append(f"\n\n# Chain of Thought:\n{self.config.cot_prompt.strip()}")

        return "\n".join(segments).strip()