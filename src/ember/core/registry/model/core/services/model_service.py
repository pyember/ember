from typing import Any, Optional, Union
from enum import Enum

from ember.core.registry.model.model_enum import ModelEnum, parse_model_str
from ember.core.registry.model.model_registry import ModelRegistry
from ember.core.registry.model.provider_registry.base_provider import BaseProviderModel
from ember.core.registry.model.core.services.usage_service import UsageService


class ModelService:
    """High-level facade for retrieving and invoking models by their identifier.

    This service integrates with a ModelRegistry to fetch model instances and with a UsageService
    to log usage statistics, if present, from model responses.

    Attributes:
        _registry (ModelRegistry): Registry that stores and provides model objects.
        _usage_service (UsageService): Service responsible for recording and managing usage records.
        _default_model_id (Optional[Union[str, Enum]]): Optional default model identifier used when no model_id is provided.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        usage_service: UsageService,
        default_model_id: Optional[Union[str, Enum]] = None,
    ) -> None:
        """Initializes ModelService with a registry, usage service, and an optional default model identifier.

        Args:
            registry (ModelRegistry): An instance of ModelRegistry for retrieving model objects.
            usage_service (UsageService): An instance of UsageService for logging model usage statistics.
            default_model_id (Optional[Union[str, Enum]]): A default model identifier used when none is provided.
        """
        self._registry: ModelRegistry = registry
        self._usage_service: UsageService = usage_service
        self._default_model_id: Optional[Union[str, Enum]] = default_model_id

    def get_model(
        self, model_id: Optional[Union[str, ModelEnum]] = None
    ) -> BaseProviderModel:
        """Retrieves a model instance from the registry using the provided identifier.

        If model_id is None, the default model identifier is used. For a ModelEnum value, its
        'value' attribute is extracted first. The identifier is then validated via parse_model_str;
        if validation fails, the raw identifier is retained. A ValueError is raised if a corresponding
        model is not found.

        Args:
            model_id (Optional[Union[str, ModelEnum]]): A string or ModelEnum representing the model identifier.

        Returns:
            BaseProviderModel: The model instance corresponding to the validated identifier.

        Raises:
            ValueError: If neither a model_id nor a default_model_id is provided, or if the model is not found.
        """
        if model_id is None:
            if self._default_model_id is None:
                raise ValueError("No model_id provided and no default_model_id set.")
            model_id = self._default_model_id

        raw_id: str = (
            model_id.value if isinstance(model_id, ModelEnum) else model_id
        )  # Explicit extraction for ModelEnum.
        try:
            validated_id: str = parse_model_str(raw_id)
        except ValueError:
            validated_id = raw_id

        model: Optional[BaseProviderModel] = self._registry.get_model(validated_id)
        if model is None:
            raise ValueError(f"Model '{validated_id}' not found.")
        return model

    def invoke_model(
        self,
        model_id: Optional[Union[str, ModelEnum]] = None,
        prompt: str = "",
        **kwargs: Any,
    ) -> Any:
        """Invokes the specified model with the given prompt and additional arguments, logging usage if available.

        This method retrieves the model instance via get_model, invokes it using named arguments,
        and subsequently logs any usage metadata present in the response.

        Args:
            model_id (Optional[Union[str, ModelEnum]]): A model identifier. Defaults to the default_model_id if None.
            prompt (str): The input prompt or query for the model.
            **kwargs (Any): Additional keyword arguments to pass to the model invocation.

        Returns:
            Any: The response from the model invocation.
        """
        model: BaseProviderModel = self.get_model(model_id=model_id)
        response: Any = model.__call__(
            prompt=prompt, **kwargs
        )  # Explicit named method invocation.

        usage = getattr(response, "usage", None)
        if usage is not None:
            self._usage_service.add_usage_record(
                model_id=model.model_info.model_id, usage_stats=usage
            )
        return response

    def __call__(
        self,
        model_id: Optional[Union[str, Enum]] = None,
        prompt: str = "",
        **kwargs: Any,
    ) -> Any:
        """Enables the ModelService instance to be called as a function, delegating to invoke_model.

        Args:
            model_id (Optional[Union[str, ModelEnum]]): A model identifier. Defaults to default_model_id if None.
            prompt (str): The user prompt or query.
            **kwargs (Any): Additional keyword arguments for the model invocation.

        Returns:
            Any: The result of the model invocation.
        """
        return self.invoke_model(model_id=model_id, prompt=prompt, **kwargs)
