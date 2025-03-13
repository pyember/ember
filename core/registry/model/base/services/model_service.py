import logging
from enum import Enum
import asyncio
from typing import Any, Optional, Union

from ember.core.registry.model.config.model_enum import (
    ModelEnum,
    parse_model_str,
)
from ember.core.registry.model.base.registry.model_registry import ModelRegistry
from ember.core.registry.model.providers.base_provider import BaseProviderModel
from ember.core.registry.model.base.services.usage_service import UsageService
from ember.core.registry.model.base.schemas.chat_schemas import ChatResponse
from ember.core.exceptions import ProviderAPIError


class ModelService:
    """Facade for retrieving and invoking models by their identifier.

    This high-level service integrates with a ModelRegistry to obtain model instances and,
    when available, logs usage statistics via a UsageService.

    Attributes:
        _registry (ModelRegistry): Registry that stores and provides model objects.
        _usage_service (Optional[UsageService]): Service responsible for recording usage data.
        _default_model_id (Optional[Union[str, Enum]]): Default identifier for models if none is supplied.
        _logger (logging.Logger): Logger instance for capturing application events.
    """

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        usage_service: Optional[UsageService] = None,
        default_model_id: Optional[Union[str, Enum]] = None,
        logger: Optional[logging.Logger] = None,
        metrics: Optional[dict[str, object]] = None,
    ) -> None:
        """
        Initializes the ModelService.

        Args:
            registry (ModelRegistry): The registry instance used to retrieve model objects.
            usage_service (Optional[UsageService]): Service to record model usage statistics.
            default_model_id (Optional[Union[str, Enum]]): Default model identifier when none is provided.
            logger (Optional[logging.Logger]): Logger instance; if not supplied, a default logger
                named after the class is created.
            metrics (Optional[dict[str, object]]): A dictionary of metrics (e.g., Prometheus counters/histograms).
        """
        self._registry: ModelRegistry = registry
        self._usage_service: Optional[UsageService] = usage_service
        self._default_model_id: Optional[Union[str, Enum]] = default_model_id
        self._logger: logging.Logger = logger or logging.getLogger(
            self.__class__.__name__
        )
        self._metrics: dict[str, object] = metrics or {}

    def get_model(
        self, model_id: Optional[Union[str, Enum]] = None
    ) -> BaseProviderModel:
        """Retrieves the model instance corresponding to the given identifier.

        If no model_id is provided, the default model identifier is used.
        For a ModelEnum value, its `value` attribute is extracted.
        The identifier is validated via `parse_model_str`; if validation fails, the raw identifier is retained.

        Args:
            model_id (Optional[Union[str, Enum]]): A string or ModelEnum representing the model identifier.
                If not provided, the default model identifier is used.

        Returns:
            BaseProviderModel: The model instance matching the validated identifier.

        Raises:
            ValueError: If neither a model identifier nor a default is provided, or if the model is not found.
        """
        if model_id is None:
            if self._default_model_id is None:
                raise ValueError("No model_id provided and no default_model_id set.")
            model_id = self._default_model_id

        raw_id: str = model_id.value if isinstance(model_id, ModelEnum) else model_id
        try:
            validated_id: str = parse_model_str(raw_id)
        except ValueError:
            validated_id = raw_id

        model: Optional[BaseProviderModel] = self._registry.get_model(validated_id)
        if model is None:
            raise ValueError(f"Model '{validated_id}' not found.")
        return model

    def invoke_model(self, model_id: str, prompt: str, **kwargs: Any) -> ChatResponse:
        metric_histogram = self._metrics.get("invocation_duration")
        # Time the model invocation using the provided histogram.
        if metric_histogram is not None:
            with metric_histogram.labels(model_id=model_id).time():
                response = self._invoke(model_id, prompt, **kwargs)
        else:
            response = self._invoke(model_id, prompt, **kwargs)
        return response

    def _invoke(self, model_id: str, prompt: str, **kwargs: Any) -> ChatResponse:
        model = self.get_model(model_id=model_id)
        try:
            response = model(prompt=prompt, **kwargs)
        except Exception as exc:
            self._logger.exception("Error invoking model '%s'.", model_id)
            raise ProviderAPIError(f"Error invoking model {model_id}") from exc

        metric_counter = self._metrics.get("model_invocations")
        if metric_counter is not None:
            metric_counter.labels(model_id=model_id).inc()

        if self._usage_service is not None and response.usage is not None:
            self._usage_service.add_usage_record(
                model_id=model.model_info.id, usage_stats=response.usage
            )
        return response

    async def invoke_model_async(
        self, model_id: str, prompt: str, **kwargs: Any
    ) -> ChatResponse:
        """Asynchronously invokes a model using the specified identifier, prompt, and additional parameters.

        If the model's invocation is implemented as a coroutine, it is awaited directly;
        otherwise, the model is executed in a separate thread.

        Args:
            model_id (str): Identifier of the model to invoke.
            prompt (str): Prompt string to be passed to the model.
            **kwargs (Any): Additional keyword arguments for the model invocation.

        Returns:
            ChatResponse: The response generated by the model.

        Raises:
            ProviderAPIError: If an error occurs during asynchronous invocation.
        """
        model: BaseProviderModel = self.get_model(model_id=model_id)
        try:
            # Checking whether the model's __call__ is asynchronous.
            if asyncio.iscoroutinefunction(model.__call__):
                response = await model(prompt=prompt, **kwargs)
            else:
                response: ChatResponse = await asyncio.to_thread(
                    model, prompt=prompt, **kwargs
                )
        except Exception as exc:
            self._logger.exception("Async error invoking model '%s'.", model_id)
            raise ProviderAPIError(
                f"Async error invoking model {model_id} with prompt '{prompt}'"
            ) from exc

        if (
            self._usage_service is not None
            and getattr(response, "usage", None) is not None
        ):
            self._usage_service.add_usage_record(
                model_id=model.model_info.id, usage_stats=response.usage
            )
        return response

    # Aliases for method invocation.
    forward = invoke_model
    __call__ = invoke_model
