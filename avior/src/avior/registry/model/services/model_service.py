from typing import Optional, Union

from src.avior.registry.model.registry.model_registry import ModelRegistry
from src.avior.registry.model.registry.model_enum import parse_model_str
from src.avior.registry.model.provider_registry.base import BaseProviderModel
from src.avior.registry.model.services.usage_service import UsageService
from src.avior.registry.model.registry.model_enum import ModelEnum

class ModelService:
    """
    High-level façade for retrieving and invoking models by a string ID or enum-based ID.
    Integrates with UsageService to record usage statistics, if provided by the model response.

    Responsibilities:
    - Validate/parse the model_id (e.g., from enum or string).
    - Fetch the model instance from the ModelRegistry.
    - Invoke the model (via __call__).
    - Log usage if the model's response includes usage metadata.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        usage_service: UsageService,
        default_model_id: Optional[Union[str, ModelEnum]] = None,
    ):
        """
        :param registry: A ModelRegistry instance that stores and provides model objects.
        :param usage_service: A UsageService for logging usage stats from model responses.
        :param default_model_id: An optional default model ID to use if no model ID is provided.
        """
        self._registry = registry
        self._usage_service = usage_service
        self._default_model_id = default_model_id

    def get_model(
        self, model_id: Optional[Union[str, "ModelEnum"]] = None
    ) -> BaseProviderModel:
        """
        Retrieves a model instance from the registry by its ID or enum. If an enum is passed in,
        we parse out its .value. If parse_model_str raises an error, we fall back to raw.
        If model_id is None, we use the default_model_id.

        :param model_id: The string ID or a ModelEnum instance representing the model.
        :return: The BaseProviderModel from the registry.
        :raises ValueError: If the model_id is not found in the registry.
        :raises ValueError: If no model_id is provided and no default_model_id is set.
        """
        if not model_id:
            if not self._default_model_id:
                raise ValueError(
                    "No model_id provided and no default_model_id set."
                )
            model_id = self._default_model_id

        raw_id = getattr(model_id, "value", model_id)
        try:
            validated_id = parse_model_str(raw_id)
        except ValueError:
            validated_id = raw_id

        model = self._registry.get_model(validated_id)
        if not model:
            raise ValueError(f"Model '{validated_id}' not found.")
        return model

    def invoke_model(
        self,
        model_id: Optional[Union[str, "ModelEnum"]] = None,
        prompt: str = "",
        **kwargs,
    ):
        """
        Invokes the model using its __call__ interface.
        If response contains usage, record it. 
        """
        model = self.get_model(model_id)
        response = model(prompt=prompt, **kwargs)

        if hasattr(response, "usage") and response.usage:
            self._usage_service.add_usage_record(
                model_id=model.model_info.model_id,
                usage_stats=response.usage,
            )
        return response

    def __call__(
        self,
        model_id: Optional[Union[str, "ModelEnum"]] = None,
        prompt: str = "",
        **kwargs,
    ):
        """
        Allows the ModelService itself to be called like a function, e.g.:
            service("openai:gpt-4o", "Hello world!")
        This is a convenience wrapper around invoke_model(...).
        If model_id is None, we use the default_model_id.

        :param model_id: Model ID or enum.
        :param prompt: The user prompt or query.
        :param kwargs: Additional parameters to pass through to the model call.
        :return: The model's response object.
        """
        return self.invoke_model(model_id, prompt, **kwargs)