"""Thin convenience layer for invoking registered language models.

The module exposes a single callable entry point along with helpers for model
discovery and reusable bindings. Configuration is inferred from the active
Ember context, so most workloads require no setup beyond calling the API.

Examples:
    >>> from ember.api import models
    >>> text = models("gpt-4", "Hello world")
    >>> text
    'Hello! How can I help you today?'
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Dict, Generator, List, Optional

from ember._internal.exceptions import ModelNotFoundError
from ember.models import ModelRegistry
from ember.models.catalog import (
    get_model_info,
    get_providers,
    list_available_models,
)
from ember.models.schemas import ChatResponse


class Response:
    """Structured wrapper around provider responses.

    Attributes:
        text: Model-generated text.
        usage: Token accounting information.
        model_id: Identifier for the model that produced the result.

    Examples:
        >>> reply = models.response("gpt-4", "What is 2 + 2?")
        >>> reply.text
        'The answer is 4.'
    """

    def __init__(self, raw_response: ChatResponse):
        """Wrap the registry response object.

        Args:
            raw_response: Provider-neutral chat response returned by the model
                registry.
        """
        self._raw = raw_response

    @property
    def chat_response(self) -> ChatResponse:
        """Expose the underlying ChatResponse instance."""

        return self._raw

    @property
    def text(self) -> str:
        """Return the primary text payload."""
        return self._raw.text

    @property
    def usage(self) -> Dict[str, Any]:
        """Return token accounting information as a dictionary."""
        if not self._raw.usage:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "actual_cost_usd": None,
                "details": None,
            }

        usage = self._raw.usage
        details: Dict[str, Any] | None = None
        if getattr(usage, "details", None):
            details = dict(usage.details)  # type: ignore[arg-type]
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "cost": (usage.cost_usd or 0.0),
            "actual_cost_usd": usage.actual_cost_usd,
            "details": details,
        }

    @property
    def model_id(self) -> Optional[str]:
        """Return the identifier of the producing model when available."""
        if hasattr(self._raw, "model_id"):
            model_id = getattr(self._raw, "model_id", None)
            if isinstance(model_id, str):
                return model_id
        if (
            hasattr(self._raw, "raw_output")
            and self._raw.raw_output is not None
            and hasattr(self._raw.raw_output, "model")
        ):
            model = getattr(self._raw.raw_output, "model", None)
            if isinstance(model, str):
                return model
        return None

    @property
    def created_at(self) -> datetime:
        """Return the UTC timestamp recorded by the provider wrapper."""

        value = getattr(self._raw, "created_at", None)
        if isinstance(value, datetime):
            return value
        return datetime.now(UTC)

    @property
    def latency_ms(self) -> Optional[float]:
        """Return the measured wall-clock latency in milliseconds."""

        return getattr(self._raw, "latency_ms", None)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Aggregate lightweight invocation metadata."""

        created_at = self.created_at
        metadata: Dict[str, Any] = {
            "model_id": self.model_id,
            "response_id": getattr(self._raw, "response_id", None),
            "created_at": (
                created_at.isoformat() if isinstance(created_at, datetime) else created_at
            ),
            "latency_ms": self.latency_ms,
        }
        # Drop keys with None values to keep the payload compact.
        return {key: value for key, value in metadata.items() if value is not None}

    @property
    def output(self) -> List[Dict[str, Any]]:
        """Return the structured output items produced by the provider."""

        structured = getattr(self._raw, "output", None)
        if isinstance(structured, list):
            return structured
        return []

    @property
    def raw(self) -> Any:
        """Expose the raw provider response for advanced consumers."""

        return getattr(self._raw, "raw_output", None)

    @property
    def raw_output(self) -> Any:
        """Backward-compatible alias for ``raw`` consumers."""

        return self.raw

    @property
    def thinking_trace(self) -> Optional[str]:
        """Return the extended thinking/reasoning trace when available.

        For models with extended thinking enabled (e.g., Claude with thinking
        configuration), this property returns the internal reasoning trace
        that the model produced before generating the final response.

        Returns:
            The thinking trace text, or None if not available.

        Examples:
            >>> reply = models.response("claude-opus-4-5-20251101", "Solve 2+2",
            ...                         thinking={"type": "enabled", "budget_tokens": 4096})
            >>> if reply.thinking_trace:
            ...     print("Model reasoning:", reply.thinking_trace[:100])
        """
        return getattr(self._raw, "thinking_trace", None)

    def __str__(self) -> str:
        """Return the generated text."""
        return self.text

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        text_preview = self.text[:50]
        total_tokens = self.usage["total_tokens"]
        return f"Response(text='{text_preview}...', tokens={total_tokens})"


@dataclass(frozen=True)
class ModelInvocation:
    """Composite result exposing text alongside the structured response."""

    text: str
    response: Response

    @property
    def usage(self) -> Dict[str, Any]:
        """Proxy usage details from the wrapped response."""

        return self.response.usage

    @property
    def model_id(self) -> Optional[str]:
        """Proxy resolved model identifier from the wrapped response."""

        return self.response.model_id

    @property
    def latency_ms(self) -> Optional[float]:
        """Proxy measured latency from the wrapped response."""

        return self.response.latency_ms

    @property
    def metadata(self) -> Dict[str, Any]:
        """Expose invocation metadata aggregated by the response wrapper."""

        return self.response.metadata


class ModelBinding:
    """Callable that reuses a model id and default parameters.

    Examples:
        >>> creative = models.instance("gpt-4", temperature=0.9)
        >>> creative("Write about dragons")
    """

    def __init__(self, model_id: str, registry: ModelRegistry, **params: Any):
        """Capture the model id, registry, and default parameters.

        Args:
            model_id: Identifier understood by the registry.
            registry: Registry that performs invocations.
            **params: Default keyword arguments forwarded to the registry.
        """
        self.model_id = model_id
        self.registry = registry
        self.params = params
        self._validate_model_id()

    def _validate_model_id(self) -> None:
        """Ensure the bound model id exists in the registry.

        Raises:
            ModelNotFoundError: The model id is unknown.
        """
        try:
            self.registry.get_model(model_id=self.model_id)
        except Exception as e:
            if "not found" in str(e).lower():
                raise ModelNotFoundError(
                    f"Model '{self.model_id}' not found",
                    context={"model_id": self.model_id},
                ) from e
            raise

    def _invoke(self, prompt: str, override_params: Dict[str, Any]) -> Response:
        """Internal helper returning the structured response instance."""

        stream_requested = bool(override_params.pop("stream", False))
        if stream_requested:
            raise ValueError("ModelBinding.stream() must be used for streaming invocations")

        merged_params = {**self.params, **override_params}
        raw_response = self.registry.invoke_model(self.model_id, prompt, **merged_params)
        return Response(raw_response)

    def __call__(self, prompt: str, **override_params: Any) -> str:
        """Invoke the binding and return the generated text."""
        stream_requested = bool(override_params.pop("stream", False))
        if stream_requested:
            raise ValueError("Use ModelBinding.stream() for streaming invocations")
        return self._invoke(prompt, override_params).text

    def response(self, prompt: str, **override_params: Any) -> Response:
        """Invoke the binding and return the structured response object."""

        return self._invoke(prompt, override_params)

    def with_metadata(self, prompt: str, **override_params: Any) -> ModelInvocation:
        """Invoke the binding and expose text plus response metadata."""

        resp = self._invoke(prompt, override_params)
        return ModelInvocation(text=resp.text, response=resp)

    def stream(self, prompt: str, **override_params: Any) -> Generator[str, None, ChatResponse]:
        """Stream the model response, yielding text chunks as they arrive."""

        merged_params = {**self.params, **override_params}
        merged_params.pop("stream", None)
        return self.registry.stream_model(self.model_id, prompt, **merged_params)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return f"ModelBinding(model_id='{self.model_id}', params={self.params})"


class ModelResult(str):
    """String-like result that retains the structured response object."""

    __slots__ = ("_response",)
    _response: Response

    def __new__(cls, response: Response) -> "ModelResult":
        instance = super().__new__(cls, response.text)
        instance._response = response
        return instance

    @property
    def text(self) -> str:
        return str(self)

    @property
    def response(self) -> Response:
        return self._response

    @property
    def usage(self) -> Dict[str, Any]:
        return self._response.usage

    @property
    def model_id(self) -> Optional[str]:
        return self._response.model_id

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._response.metadata

    def __repr__(self) -> str:
        return f"ModelResult(text={str(self)!r})"

    @property
    def output(self) -> List[Dict[str, Any]]:
        return self._response.output

    @property
    def raw(self) -> Any:
        return self._response.raw

    @property
    def thinking_trace(self) -> Optional[str]:
        return self._response.thinking_trace


class ModelsAPI:
    """Context-aware facade that powers the public ``models`` callable.

    Examples:
        >>> models("gpt-4", "Hello world")  # doctest: +SKIP
        'Hello! How can I help you today?'
    """

    def __init__(self) -> None:
        """Acquire the shared context and registry lazily."""
        from ember._internal.context import EmberContext

        # Get or create context
        self._context = EmberContext.current()

        # Get registry from context (lazy initialization)
        self._registry = self._context.model_registry

    def _invoke(self, model: str, prompt: str, **params: Any) -> Response:
        """Internal helper returning the structured response wrapper."""

        stream_requested = bool(params.pop("stream", False))
        if stream_requested:
            raise ValueError("Use ModelsAPI.stream() for streaming invocations")

        raw_response = self._registry.invoke_model(model, prompt, **params)
        return Response(raw_response)

    def _build_responses_payload(
        self,
        payload: Optional[Mapping[str, Any]],
        *,
        input: object | None,
        instructions: object | None,
        messages: Sequence[Mapping[str, Any]] | None,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize keyword arguments into a Responses API payload."""

        request_payload: Dict[str, Any] = {}
        if payload is not None:
            if not isinstance(payload, Mapping):
                raise TypeError("payload must be a mapping when provided")
            request_payload.update(copy.deepcopy(dict(payload)))

        if input is not None:
            request_payload["input"] = input

        if instructions is not None:
            request_payload["instructions"] = instructions

        if messages is not None:
            normalized_messages: list[dict[str, Any]] = []
            for index, message in enumerate(messages):
                if not isinstance(message, Mapping):
                    raise TypeError(f"messages entry {index} must be a mapping")
                normalized_messages.append(dict(message))
            request_payload["messages"] = normalized_messages

        prompt = params.pop("prompt", None)
        if prompt is not None:
            if "input" in request_payload and request_payload["input"] is not None:
                raise ValueError("Cannot supply both 'prompt' and 'input'")
            request_payload["input"] = prompt

        context = params.pop("context", None)
        if context is not None and "messages" not in request_payload:
            if not isinstance(context, Sequence):
                raise TypeError("context must be a sequence when supplied")
            normalized_context: list[dict[str, Any]] = []
            for index, message in enumerate(context):
                if not isinstance(message, Mapping):
                    raise TypeError(f"context entry {index} must be a mapping")
                normalized_context.append(dict(message))
            if normalized_context:
                request_payload["messages"] = normalized_context

        max_tokens = params.pop("max_tokens", None)
        if max_tokens is not None and "max_output_tokens" not in request_payload:
            request_payload["max_output_tokens"] = max_tokens

        params.pop("stream", None)

        for key, value in list(params.items()):
            if value is None:
                continue
            request_payload[key] = value

        # Drop keys with None values to keep payload compact
        return {k: v for k, v in request_payload.items() if v is not None}

    def __call__(self, model: str, prompt: str, **params: Any) -> ModelResult:
        """Invoke a model via the registry using the active context.

        Args:
            model: Model identifier understood by the registry.
            prompt: Prompt text to send.
            **params: Additional provider parameters such as ``temperature`` or
                ``max_tokens``.

        Returns:
            ModelResult: String-like result exposing the underlying ``Response``.

        Raises:
            ModelNotFoundError: The model id is unknown.
            ModelProviderError: The provider lacks authentication material.
            ProviderAPIError: The provider returned an error response.

        Examples:
            >>> api = ModelsAPI()
            >>> api("gpt-4", "Ping").text  # doctest: +SKIP
            'Pong'
        """
        stream_requested = bool(params.pop("stream", False))
        if stream_requested:
            raise ValueError("Use models.stream() for streaming invocations")
        return ModelResult(self._invoke(model, prompt, **params))

    def response(self, model: str, prompt: str, **params: Any) -> Response:
        """Invoke and return the structured response object."""

        return self._invoke(model, prompt, **params)

    def with_metadata(self, model: str, prompt: str, **params: Any) -> ModelInvocation:
        """Invoke and expose both text and response metadata."""

        resp = self._invoke(model, prompt, **params)
        return ModelInvocation(text=resp.text, response=resp)

    def instance(self, model: str, **params: Any) -> ModelBinding:
        """Return a :class:`ModelBinding` with default parameters applied.

        Args:
            model: Model identifier.
            **params: Default keyword arguments for future invocations.

        Returns:
            ModelBinding: Callable object that preserves the defaults.

        Examples:
            >>> generator = models.instance("gpt-4", temperature=0.8)
            >>> generator("Write a limerick")
        """
        return ModelBinding(model, self._registry, **params)

    def stream(self, model: str, prompt: str, **params: Any) -> Generator[str, None, ChatResponse]:
        """Stream a model response, yielding text chunks until completion."""

        params.pop("stream", None)
        return self._registry.stream_model(model, prompt, **params)

    def responses(
        self,
        model: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        input: object | None = None,
        instructions: object | None = None,
        messages: Sequence[Mapping[str, Any]] | None = None,
        **params: Any,
    ) -> Response:
        """Invoke a model using the Responses API surface."""

        stream_requested = bool(params.pop("stream", False))
        if stream_requested:
            raise ValueError("Use ModelsAPI.responses_stream() for streaming invocations")

        payload_dict = self._build_responses_payload(
            payload,
            input=input,
            instructions=instructions,
            messages=messages,
            params=dict(params),
        )
        raw_response = self._registry.invoke_responses(model, payload_dict)
        return Response(raw_response)

    def responses_stream(
        self,
        model: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        input: object | None = None,
        instructions: object | None = None,
        messages: Sequence[Mapping[str, Any]] | None = None,
        **params: Any,
    ) -> Generator[str, None, ChatResponse]:
        """Stream a Responses API invocation."""

        params.pop("stream", None)
        payload_dict = self._build_responses_payload(
            payload,
            input=input,
            instructions=instructions,
            messages=messages,
            params=dict(params),
        )
        return self._registry.stream_responses(model, payload_dict)

    def list(
        self,
        provider: Optional[str] = None,
        *,
        include_dynamic: bool = True,
        refresh: bool = False,
        discovery_mode: Optional[str] = None,
    ) -> List[str]:
        """Return the sorted list of known model identifiers.

        Args:
            provider: Optional provider filter to scope results.
            include_dynamic: When ``True``, include models discovered via live APIs.
            refresh: Force a refresh of dynamic discovery caches.

        Examples:
            >>> models.list()[:2]  # doctest: +SKIP
            ['gpt-4', 'gpt-4-turbo']
        """
        return list_available_models(
            provider=provider,
            include_dynamic=include_dynamic,
            refresh=refresh,
            discovery_mode=discovery_mode,
        )

    def discover(
        self,
        provider: Optional[str] = None,
        *,
        include_dynamic: bool = True,
        refresh: bool = False,
        discovery_mode: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return detailed metadata for available models.

        Args:
            provider: Optional provider slug used to filter results.

        Returns:
            dict[str, dict[str, Any]]: Mapping of model id to metadata such as
            provider, description, and context window size.

        Examples:
            >>> models.discover("openai").keys()
            dict_keys([...])
        """
        metadata: Dict[str, Dict[str, Any]] = {}
        for model_id in list_available_models(
            provider,
            include_dynamic=include_dynamic,
            refresh=refresh,
            discovery_mode=discovery_mode,
        ):
            info = get_model_info(
                model_id,
                include_dynamic=include_dynamic,
                refresh=refresh,
                discovery_mode=discovery_mode,
            )
            metadata[model_id] = {
                "provider": info.provider,
                "description": info.description,
                "context_window": info.context_window,
                "context_window_out": info.context_window_out,
                "aliases": list(info.aliases),
                "capabilities": list(info.capabilities),
                "region_scope": list(info.region_scope),
                "status": info.status,
                "hidden": info.hidden,
            }
        return metadata

    def providers(self) -> List[str]:
        """Return the list of provider identifiers exposed in the registry.

        Examples:
            >>> 'openai' in models.providers()
            True
        """
        return sorted(get_providers())

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return "ModelsAPI(simplified=True, direct_init=True)"


# Create global instance
_global_models_api = ModelsAPI()


class ModelsFacade:
    """Callable facade that forwards to :class:`ModelsAPI`."""

    def __init__(self, api: ModelsAPI) -> None:
        self._api = api

    def __call__(self, model: str, prompt: str, **params: Any) -> ModelResult:
        """Invoke a language model without instantiating a client.

        Args:
            model: Model identifier to invoke.
            prompt: Prompt text forwarded to the provider.
            **params: Additional keyword arguments understood by the provider.

        Returns:
            str: Primary text payload produced by the model.

        Examples:
            >>> models("gpt-4", "Hello world")  # doctest: +SKIP
            'Hello! How can I help you today?'
        """
        return self._api(model, prompt, **params)

    def instance(self, model: str, **params: Any) -> ModelBinding:
        return self._api.instance(model, **params)

    def response(self, model: str, prompt: str, **params: Any) -> Response:
        return self._api.response(model, prompt, **params)

    def with_metadata(self, model: str, prompt: str, **params: Any) -> ModelInvocation:
        return self._api.with_metadata(model, prompt, **params)

    def stream(self, model: str, prompt: str, **params: Any) -> Generator[str, None, ChatResponse]:
        return self._api.stream(model, prompt, **params)

    def responses(
        self,
        model: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        input: object | None = None,
        instructions: object | None = None,
        messages: Sequence[Mapping[str, Any]] | None = None,
        **params: Any,
    ) -> Response:
        return self._api.responses(
            model,
            payload=payload,
            input=input,
            instructions=instructions,
            messages=messages,
            **params,
        )

    def responses_stream(
        self,
        model: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        input: object | None = None,
        instructions: object | None = None,
        messages: Sequence[Mapping[str, Any]] | None = None,
        **params: Any,
    ) -> Generator[str, None, ChatResponse]:
        return self._api.responses_stream(
            model,
            payload=payload,
            input=input,
            instructions=instructions,
            messages=messages,
            **params,
        )

    def list(
        self,
        provider: Optional[str] = None,
        *,
        include_dynamic: bool = True,
        refresh: bool = False,
        discovery_mode: Optional[str] = None,
    ) -> List[str]:
        return self._api.list(
            provider=provider,
            include_dynamic=include_dynamic,
            refresh=refresh,
            discovery_mode=discovery_mode,
        )

    def discover(
        self,
        provider: Optional[str] = None,
        *,
        include_dynamic: bool = True,
        refresh: bool = False,
        discovery_mode: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        return self._api.discover(
            provider,
            include_dynamic=include_dynamic,
            refresh=refresh,
            discovery_mode=discovery_mode,
        )

    def providers(self) -> List[str]:
        return self._api.providers()

    def __repr__(self) -> str:
        return f"ModelsFacade(api={self._api!r})"


models = ModelsFacade(_global_models_api)
