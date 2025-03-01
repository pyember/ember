import logging
from typing import Optional, Any
from pydantic import BaseModel, Field

from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.base.services.usage_service import UsageService

logger: logging.Logger = logging.getLogger(__name__)


class LMModuleConfig(BaseModel):
    """Configuration settings for the Language Model module.

    Attributes:
        id (str): Identifier for selecting the underlying model provider.
        temperature (float): Sampling temperature for model generation.
        max_tokens (Optional[int]): Maximum tokens to generate in a single forward call.
        cot_prompt (Optional[str]): Chain-of-thought prompt appended to the user's prompt.
        persona (Optional[str]): Persona or role context prepended to the user query.
    """

    id: str = Field(
        default="openai:gpt-4o",
        description="Identifier for the underlying model provider.",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description="Sampling temperature for model generation.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate in one call.",
    )
    cot_prompt: Optional[str] = Field(
        default=None,
        description="Optional chain-of-thought prompt to append.",
    )
    persona: Optional[str] = Field(
        default=None,
        description="Optional persona context to prepend to the query.",
    )


def get_default_model_service() -> ModelService:
    """Creates and returns a default ModelService instance using the new unified initializer.

    Instead of relying on a global registry, we explicitly call initialize_ember()
    so that the registry is built from the current configuration.
    """
    from ember.core.registry.model.config.settings import initialize_ember

    # Initialize the registry (with default flags; these can be adjusted as needed).
    registry = initialize_ember(auto_register=True, auto_discover=True)
    usage_service = (
        UsageService()
    )  # Optionally, you might inject a custom usage service.
    return ModelService(registry=registry, usage_service=usage_service)


class LMModule:
    """Language Model module that integrates with ModelService and optional usage tracking.

    When the flag `simulate_api` is True, this module returns a dummy response rather than calling the real API.

    This module is designed to generate text responses based on a user prompt. It merges
    persona and chain-of-thought details into the prompt and delegates model invocation to
    the ModelService.

    Example:
        lm_config = LMModuleConfig(model_id="provider:custom-model", temperature=0.7)
        lm_module = LMModule(config=lm_config)
        response_text = lm_module(prompt="Hello, world!")
    """

    def __init__(
        self,
        config: LMModuleConfig,
        model_service: Optional[ModelService] = None,
        simulate_api: bool = False,
    ) -> None:
        """Initializes the LMModule.

        Args:
            config (LMModuleConfig): Configuration for model settings such as model_id and temperature.
            model_service (Optional[ModelService]): Service for model invocation. If None, a
                default ModelService is created.
            simulate_api (bool): Flag indicating whether to simulate API calls.
        """
        if model_service is None:
            model_service = get_default_model_service()
        self.config: LMModuleConfig = config
        self.model_service: ModelService = model_service
        self.simulate_api: bool = simulate_api
        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Enables the LMModule instance to be called like a function.

        This method simply forwards the call to the forward() method.

        Args:
            prompt (str): The input prompt to generate text from.
            **kwargs (Any): Additional keyword arguments for model invocation.

        Returns:
            str: The generated text response.
        """
        return self.forward(prompt=prompt, **kwargs)

    def forward(self, prompt: str, **kwargs: Any) -> str:
        """Generates text from a prompt by delegating to the ModelService.

        This method assembles a final prompt by merging persona information and a chain-of-thought
        prompt (if provided) with the user's prompt, then calls the ModelService to generate
        the response.

        Args:
            prompt (str): The user-provided prompt.
            **kwargs (Any): Additional parameters for model invocation (e.g., temperature, max_tokens).

        Returns:
            str: The generated text response.
            If the module is configured to simulate API calls and the flag simulate_api is True,
            returns a dummy response immediately.

        """
        if self.simulate_api:
            self._logger.debug("Simulating API call for prompt: %s", prompt)
            return f"SIMULATED_RESPONSE: {prompt}"
        final_prompt: str = self._assemble_full_prompt(user_prompt=prompt)
        response: Any = self.model_service.invoke_model(
            model_id=self.config.id,
            prompt=final_prompt,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            **kwargs,
        )
        # Return response content if available; otherwise, convert response to string.
        return response.data if hasattr(response, "data") else str(response)

    def _assemble_full_prompt(self, user_prompt: str) -> str:
        """Assembles the full prompt by merging persona and chain-of-thought details.

        Args:
            user_prompt (str): The base prompt provided by the user.

        Returns:
            str: The final prompt after merging additional context.
        """
        segments: list[str] = []
        if self.config.persona:
            segments.append(f"[Persona: {self.config.persona}]\n")
        segments.append(user_prompt.strip())
        if self.config.cot_prompt:
            segments.append(
                f"\n\n# Chain of Thought:\n{self.config.cot_prompt.strip()}"
            )
        return "\n".join(segments).strip()
