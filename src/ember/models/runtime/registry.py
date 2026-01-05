"""Thread-safe registry for resolving and invoking language models.

The registry caches provider instances, computes estimated usage costs, and exposes
sync/async invocation helpers used throughout Ember.

Examples:
    >>> registry = ModelRegistry()
    >>> isinstance(registry.list_models(), list)
    True
"""

import asyncio
import copy
import logging
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Generator, Optional, Sequence

from ember._internal.exceptions import (
    ModelNotFoundError,
    ModelProviderError,
    ProviderAPIError,
)
from ember.core.credentials import CredentialNotFoundError
from ember.models.catalog import canonicalize_model_identifier
from ember.models.pricing import get_model_cost
from ember.models.pricing.manager import PricingNotFoundError
from ember.models.pricing.tracker import track_usage
from ember.models.providers import (
    get_provider_class,
    resolve_model_id,
)
from ember.models.providers.base import BaseProvider
from ember.models.providers.utils import sanitize_responses_payload
from ember.models.schemas import ChatResponse, EmbeddingResponse, UsageStats
from ember.xcs.runtime import Reservation

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ember._internal.context.runtime import EmberContext


def _merge_nested_defaults(target: dict[str, Any], defaults: Mapping[str, Any]) -> None:
    """Apply alias defaults onto a payload without overriding explicit values."""

    for key, value in defaults.items():
        if key in target and target[key] is not None:
            existing = target[key]
            if isinstance(existing, Mapping) and isinstance(value, Mapping):
                merged = copy.deepcopy(value)
                merged.update(existing)
                target[key] = merged
            continue

        if isinstance(value, Mapping):
            target[key] = copy.deepcopy(value)
        elif isinstance(value, list):
            target[key] = list(value)
        elif isinstance(value, tuple):
            target[key] = tuple(value)
        else:
            target[key] = value


class ModelRegistry:
    """Manage provider instances and track usage for Ember models.

    The registry handles lazy instantiation, sync/async invocation, and cost tracking so
    callers can work with model identifiers without worrying about provider-specific details.

    Attributes:
        _models: Cache of instantiated provider instances keyed by model id.
        _lock: Thread synchronization primitive guarding registry state.
        _usage_records: Recorded ``UsageStats`` entries per model id.
        _metrics: Optional metric collectors keyed by metric name.

    Examples:
        >>> registry = ModelRegistry()
        >>> 'gpt-4' in registry.list_models()
        False
    """

    def __init__(
        self, metrics: Optional[dict[str, Any]] = None, context: Optional[Any] = None
    ) -> None:
        """Initialize the registry with optional metrics and context hooks.

        Args:
            metrics: Optional mapping of metric collectors keyed by metric name.
            context: Optional EmberContext used to resolve provider credentials.

        Examples:
            >>> ModelRegistry(metrics={'calls': object()})  # doctest: +SKIP
        """
        self._models: dict[str, BaseProvider] = {}
        self._lock = threading.Lock()
        self._usage_records: dict[str, list[UsageStats]] = {}
        self._metrics = metrics or {}
        self._logger = logger
        self._context = context

    def get_model(self, model_id: str) -> BaseProvider:
        """Return a cached provider instance for the requested model.

        Args:
            model_id: Model identifier (plain name or ``provider/model``).

        Returns:
            BaseProvider: Provider instance ready for invocation.

        Raises:
            ModelNotFoundError: If the model cannot be resolved.
            ModelProviderError: If credentials are missing or invalid.

        Examples:
            >>> registry = ModelRegistry()
            >>> registry.get_model('openai/gpt-4')  # doctest: +SKIP
        """
        canonical_id, _ = canonicalize_model_identifier(model_id)
        if canonical_id != model_id:
            self._logger.debug("canonicalized model id '%s' -> '%s'", model_id, canonical_id)
        model_id = canonical_id

        # Fast path: check cache without lock
        if model_id in self._models:
            return self._models[model_id]

        # Slow path: create model with lock
        with self._lock:
            # Double-check pattern after acquiring lock
            if model_id in self._models:
                return self._models[model_id]

            # Create and cache the model
            model = self._create_model(model_id)
            self._models[model_id] = model
            return model

    def _create_model(self, model_id: str) -> BaseProvider:
        """Instantiate a provider for the requested model identifier.

        Args:
            model_id: Identifier resolved via the catalog/provider mapping.

        Returns:
            BaseProvider: Newly instantiated provider.

        Raises:
            ModelNotFoundError: If the provider cannot be resolved.
            ModelProviderError: If credentials are unavailable.
        """
        # Resolve provider name from model ID
        provider_name, model_name = resolve_model_id(model_id)

        if provider_name == "unknown":
            from ember.models.catalog import list_available_models

            available = list_available_models()
            raise ModelNotFoundError(
                f"Cannot determine provider for model '{model_id}'. "
                f"Available models: {', '.join(sorted(available))}",
                context={"model_id": model_id, "available_models": available},
            )

        # Get provider implementation class
        try:
            provider_class = get_provider_class(provider_name)
        except ValueError as e:
            raise ModelNotFoundError(
                f"Provider '{provider_name}' not found",
                context={"provider": provider_name, "model_id": model_id},
            ) from e

        # If provider does not require API key, instantiate directly
        if getattr(provider_class, "requires_api_key", True) is False:
            return provider_class(api_key=None)

        context = self._context or self._resolve_context()
        if context is None:
            raise RuntimeError("EmberContext is required for credential resolution")

        try:
            api_key = context.get_credential(provider_name)
        except CredentialNotFoundError as exc:
            from ember.core.setup_launcher import launch_setup_if_needed

            api_key = launch_setup_if_needed(provider_name, model_id)
            if api_key is None:
                from ember.core.setup_launcher import format_non_interactive_error

                raise ModelProviderError(
                    format_non_interactive_error(provider_name, model_id),
                    context={"model_id": model_id, "provider": provider_name},
                ) from exc

        # Instantiate provider
        try:
            provider = provider_class(api_key=api_key)
            return provider
        except ValueError as e:
            # Handle missing API key error from BaseProvider
            if "API key required" in str(e):
                # Re-raise as ModelProviderError for better error handling
                from ember.core.setup_launcher import format_non_interactive_error

                error_msg = format_non_interactive_error(provider_name, model_id)
                raise ModelProviderError(
                    error_msg, context={"model_id": model_id, "provider": provider_name}
                ) from e
            else:
                # Other ValueError, re-raise as ModelNotFoundError
                raise ModelNotFoundError(
                    f"Failed to instantiate provider for model {model_id}",
                    context={"model_id": model_id, "provider": provider_name},
                ) from e

    def invoke_model(self, model_id: str, prompt: str, **kwargs: Any) -> ChatResponse:
        """Invoke a model synchronously and record usage/cost metrics.

        Args:
            model_id: Canonical or provider-qualified model name.
            prompt: Prompt text sent to the provider.
            **kwargs: Provider-specific options forwarded to the provider.

        Returns:
            ChatResponse: Response containing text, usage, and cost estimates.

        Raises:
            ProviderAPIError: If the provider call fails.
            ModelNotFoundError: If the model id cannot be resolved.
            ModelProviderError: If credentials are unavailable.

        Examples:
            >>> registry = ModelRegistry()
            >>> registry.invoke_model('gpt-4', 'Hello')  # doctest: +SKIP
        """
        start_wall = datetime.now(UTC)
        start_perf = time.perf_counter()
        response: ChatResponse | None = None
        requested_model_id = model_id
        model_id, alias_defaults = canonicalize_model_identifier(model_id)
        alias_context: dict[str, Any] = {}
        if model_id != requested_model_id:
            alias_context["requested_model_id"] = requested_model_id

        reservation = kwargs.pop("_reservation", None)
        if reservation is not None and not isinstance(reservation, Reservation):
            reservation = None

        for key, value in alias_defaults.items():
            if key == "reasoning" and "reasoning" in kwargs:
                continue
            if isinstance(value, dict):
                kwargs.setdefault(key, dict(value))
            else:
                kwargs.setdefault(key, value)

        # Resolve provider/model and get the model provider
        provider_name, model_name = resolve_model_id(model_id)
        self._logger.debug(
            "resolve: %s -> provider=%s, model=%s", model_id, provider_name, model_name
        )
        if reservation is not None:
            requested_model = reservation.model or model_id
            if reservation.model:
                canonical_reservation_id, reservation_defaults = canonicalize_model_identifier(
                    requested_model
                )
                if canonical_reservation_id != requested_model:
                    alias_context["reservation_model_id"] = requested_model
                    requested_model = canonical_reservation_id
                for key, value in reservation_defaults.items():
                    if key == "reasoning" and "reasoning" in kwargs:
                        continue
                    if isinstance(value, dict):
                        kwargs.setdefault(key, dict(value))
                    else:
                        kwargs.setdefault(key, value)
            provider_override, model_override = resolve_model_id(requested_model)
            if provider_override != "unknown":
                provider_name = provider_override
            elif reservation.provider:
                provider_name = reservation.provider
            if model_override:
                model_name = model_override
            model_id = requested_model
        model = self.get_model(model_id)

        # Record invocation metric if available
        if "model_invocations" in self._metrics:
            self._metrics["model_invocations"].labels(model_id=model_id).inc()

        try:
            # Invoke with optional metrics timing
            invocation_kwargs = dict(kwargs)
            if reservation is not None and reservation.credentials:
                cred_overrides = dict(reservation.credentials)
                cred_overrides.update(invocation_kwargs)
                invocation_kwargs = cred_overrides
            if "invocation_duration" in self._metrics:
                with self._metrics["invocation_duration"].labels(model_id=model_id).time():
                    response = model.complete(prompt, model_name, **invocation_kwargs)
            else:
                response = model.complete(prompt, model_name, **invocation_kwargs)

            # Calculate and add cost if usage is available
            if response.usage:
                usage = response.usage
                estimated_cost = self._calculate_cost(model_id, usage)
                actual_cost = usage.actual_cost_usd

                # Feed the reconciliation tracker with both the estimate and provider bill
                tracker_sample = UsageStats(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    cost_usd=estimated_cost,
                    actual_cost_usd=actual_cost,
                )
                track_usage(tracker_sample, model_id)

                # Surface the most authoritative number to callers
                usage.cost_usd = actual_cost if actual_cost is not None else estimated_cost

                # Track usage for summaries with the surfaced cost
                self._track_usage(model_id, usage)

            # Ensure model_id is in response
            if not response.model_id:
                response.model_id = model_id

            duration_ms = (time.perf_counter() - start_perf) * 1000.0
            response.latency_ms = duration_ms
            if response.started_at is None:
                response.started_at = start_wall
            if response.completed_at is None:
                response.completed_at = datetime.now(UTC)

            return response

        except (ProviderAPIError, ModelNotFoundError, ModelProviderError, PricingNotFoundError):
            raise
        except Exception as exc:
            self._logger.exception("Unexpected error invoking model '%s'", model_id)
            error_context = {"model_id": model_id}
            if alias_context:
                error_context.update(alias_context)
            raise ProviderAPIError(
                f"Error invoking model {model_id}",
                context=error_context,
            ) from exc
        finally:
            duration = time.perf_counter() - start_perf
            self._logger.debug(f"Model {model_id} invocation took {duration:.3f}s")

    def invoke_responses(
        self,
        model_id: str,
        payload: Mapping[str, Any],
        **kwargs: Any,
    ) -> ChatResponse:
        """Invoke a model via the Responses API surface."""

        if not isinstance(payload, Mapping):
            raise ValueError("payload must be a mapping")

        start_wall = datetime.now(UTC)
        start_perf = time.perf_counter()
        response: ChatResponse | None = None

        requested_model_id = model_id
        model_id, alias_defaults = canonicalize_model_identifier(model_id)
        alias_context: dict[str, Any] = {}
        if model_id != requested_model_id:
            alias_context["requested_model_id"] = requested_model_id

        reservation = kwargs.pop("_reservation", None)
        if reservation is not None and not isinstance(reservation, Reservation):
            reservation = None

        invocation_kwargs = dict(kwargs)

        provider_name, model_name = resolve_model_id(model_id)
        self._logger.debug(
            "responses resolve: %s -> provider=%s, model=%s",
            model_id,
            provider_name,
            model_name,
        )

        if reservation is not None:
            requested_model = reservation.model or model_id
            reservation_defaults: Mapping[str, Any] | None = None
            if reservation.model:
                canonical_reservation_id, reservation_defaults = canonicalize_model_identifier(
                    requested_model
                )
                if canonical_reservation_id != requested_model:
                    alias_context["reservation_model_id"] = requested_model
                    requested_model = canonical_reservation_id
            else:
                reservation_defaults = None

            provider_override, model_override = resolve_model_id(requested_model)
            if provider_override != "unknown":
                provider_name = provider_override
            elif reservation.provider:
                provider_name = reservation.provider
            if model_override:
                model_name = model_override
            model_id = requested_model

            if reservation_defaults:
                # reservation defaults should win when caller omitted explicit values
                invocation_payload_defaults = dict(reservation_defaults)
            else:
                invocation_payload_defaults = {}
            # Merge reservation credentials into invocation kwargs
            if reservation.credentials:
                invocation_kwargs.update(dict(reservation.credentials))
        else:
            invocation_payload_defaults = {}

        model = self.get_model(model_id)
        if not getattr(model, "supports_responses_api", False):
            raise ProviderAPIError(
                "Provider does not support Responses API",
                context={"model_id": model_id, "provider": provider_name},
            )

        if "model_invocations" in self._metrics:
            self._metrics["model_invocations"].labels(model_id=model_id).inc()

        invocation_payload = copy.deepcopy(dict(payload))
        canonical_model_name = model_name or model_id
        invocation_payload["model"] = canonical_model_name
        _merge_nested_defaults(invocation_payload, alias_defaults)
        if invocation_payload_defaults:
            _merge_nested_defaults(invocation_payload, invocation_payload_defaults)
        invocation_payload = sanitize_responses_payload(canonical_model_name, invocation_payload)

        try:
            response = model.complete_responses_payload(invocation_payload, **invocation_kwargs)

            if response.usage:
                usage = response.usage
                estimated_cost = self._calculate_cost(model_id, usage)
                actual_cost = usage.actual_cost_usd

                tracker_sample = UsageStats(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    cost_usd=estimated_cost,
                    actual_cost_usd=actual_cost,
                    details=usage.details,
                )
                track_usage(tracker_sample, model_id)

                usage.cost_usd = actual_cost if actual_cost is not None else estimated_cost
                self._track_usage(model_id, usage)

            if not response.model_id:
                response.model_id = model_id
            response.latency_ms = (time.perf_counter() - start_perf) * 1000.0
            if response.started_at is None:
                response.started_at = start_wall
            if response.completed_at is None:
                response.completed_at = datetime.now(UTC)

            return response
        except (ProviderAPIError, ModelNotFoundError, ModelProviderError, PricingNotFoundError):
            raise
        except Exception as exc:
            self._logger.exception("Error invoking responses model '%s'", model_id)
            error_context = {"model_id": model_id}
            if alias_context:
                error_context.update(alias_context)
            raise ProviderAPIError(
                f"Error invoking model {model_id} via Responses API",
                context=error_context,
            ) from exc
        finally:
            duration = time.perf_counter() - start_perf
            self._logger.debug(f"Model {model_id} responses invocation took {duration:.3f}s")

    def stream_responses(
        self,
        model_id: str,
        payload: Mapping[str, Any],
        **kwargs: Any,
    ) -> Generator[str, None, ChatResponse]:
        """Stream a Responses API invocation."""

        if not isinstance(payload, Mapping):
            raise ValueError("payload must be a mapping")

        start_wall = datetime.now(UTC)
        start_perf = time.perf_counter()
        requested_model_id = model_id
        model_id, alias_defaults = canonicalize_model_identifier(model_id)
        alias_context: dict[str, Any] = {}
        if model_id != requested_model_id:
            alias_context["requested_model_id"] = requested_model_id

        reservation = kwargs.pop("_reservation", None)
        if reservation is not None and not isinstance(reservation, Reservation):
            reservation = None

        invocation_kwargs = dict(kwargs)

        provider_name, model_name = resolve_model_id(model_id)
        self._logger.debug(
            "responses stream resolve: %s -> provider=%s, model=%s",
            model_id,
            provider_name,
            model_name,
        )

        invocation_payload_defaults: Mapping[str, Any] | None = None
        if reservation is not None:
            requested_model = reservation.model or model_id
            if reservation.model:
                canonical_reservation_id, invocation_payload_defaults = (
                    canonicalize_model_identifier(requested_model)
                )
                if canonical_reservation_id != requested_model:
                    alias_context["reservation_model_id"] = requested_model
                    requested_model = canonical_reservation_id
            provider_override, model_override = resolve_model_id(requested_model)
            if provider_override != "unknown":
                provider_name = provider_override
            elif reservation.provider:
                provider_name = reservation.provider
            if model_override:
                model_name = model_override
            model_id = requested_model

            if reservation.credentials:
                invocation_kwargs.update(dict(reservation.credentials))
        else:
            invocation_payload_defaults = None

        model = self.get_model(model_id)
        if not getattr(model, "supports_responses_api", False):
            raise ProviderAPIError(
                "Provider does not support Responses API streaming",
                context={"model_id": model_id, "provider": provider_name},
            )

        if "model_invocations" in self._metrics:
            self._metrics["model_invocations"].labels(model_id=model_id).inc()

        invocation_payload = copy.deepcopy(dict(payload))
        canonical_model_name = model_name or model_id
        invocation_payload["model"] = canonical_model_name
        _merge_nested_defaults(invocation_payload, alias_defaults)
        if invocation_payload_defaults:
            _merge_nested_defaults(invocation_payload, invocation_payload_defaults)

        def _stream() -> Generator[str, None, ChatResponse]:
            try:
                sanitized_payload = sanitize_responses_payload(
                    canonical_model_name,
                    invocation_payload,
                )
                stream = model.stream_responses_payload(sanitized_payload, **invocation_kwargs)
                while True:
                    chunk = next(stream)
                    yield chunk
            except StopIteration as stop:
                response = stop.value
                if not isinstance(response, ChatResponse):
                    raise ProviderAPIError(
                        "Streaming provider must return ChatResponse",
                        context={"model_id": model_id},
                    ) from stop

                if not response.model_id:
                    response.model_id = model_id

                if response.started_at is None:
                    response.started_at = start_wall
                if response.completed_at is None:
                    response.completed_at = datetime.now(UTC)
                response.latency_ms = (time.perf_counter() - start_perf) * 1000.0

                if response.usage:
                    usage = response.usage
                    estimated_cost = self._calculate_cost(model_id, usage)
                    actual_cost = usage.actual_cost_usd

                    tracker_sample = UsageStats(
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        total_tokens=usage.total_tokens,
                        cost_usd=estimated_cost,
                        actual_cost_usd=actual_cost,
                        details=usage.details,
                    )
                    track_usage(tracker_sample, model_id)

                    usage.cost_usd = actual_cost if actual_cost is not None else estimated_cost
                    self._track_usage(model_id, usage)

                return response
            except ProviderAPIError:
                raise
            except Exception as e:
                self._logger.exception(f"Error streaming responses model '{model_id}'")
                error_context = {"model_id": model_id}
                if alias_context:
                    error_context.update(alias_context)
                raise ProviderAPIError(
                    f"Error streaming model {model_id} via Responses API", context=error_context
                ) from e
            finally:
                duration = time.perf_counter() - start_perf
                self._logger.debug(f"Model {model_id} responses streaming took {duration:.3f}s")

        return _stream()

    def embed_model(self, model_id: str, inputs: Sequence[str], **kwargs: Any) -> EmbeddingResponse:
        """Generate embeddings for the supplied inputs via the requested model."""

        if not inputs:
            raise ValueError("inputs must contain at least one string")
        if not all(isinstance(item, str) for item in inputs):
            raise TypeError("All embedding inputs must be strings")

        start_wall = datetime.now(UTC)
        start_perf = time.perf_counter()
        requested_model_id = model_id
        model_id, alias_defaults = canonicalize_model_identifier(model_id)
        alias_context: dict[str, Any] = {}
        if model_id != requested_model_id:
            alias_context["requested_model_id"] = requested_model_id

        for key, value in alias_defaults.items():
            if isinstance(value, dict):
                kwargs.setdefault(key, dict(value))
            else:
                kwargs.setdefault(key, value)

        provider_name, model_name = resolve_model_id(model_id)
        self._logger.debug(
            "embed resolve: %s -> provider=%s, model=%s", model_id, provider_name, model_name
        )

        model = self.get_model(model_id)

        if "model_invocations" in self._metrics:
            self._metrics["model_invocations"].labels(model_id=model_id).inc()

        try:
            response = model.embed(list(inputs), model_name, **kwargs)

            if response.usage:
                usage = response.usage
                estimated_cost = self._calculate_cost(model_id, usage)
                actual_cost = usage.actual_cost_usd

                tracker_sample = UsageStats(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    cost_usd=estimated_cost,
                    actual_cost_usd=actual_cost,
                )
                track_usage(tracker_sample, model_id)
                usage.cost_usd = actual_cost if actual_cost is not None else estimated_cost
                self._track_usage(model_id, usage)

            if not response.model_id:
                response.model_id = model_id
            response.latency_ms = (time.perf_counter() - start_perf) * 1000.0
            if response.started_at is None:
                response.started_at = start_wall
            if response.completed_at is None:
                response.completed_at = datetime.now(UTC)

            return response

        except ProviderAPIError:
            raise
        except Exception as exc:
            self._logger.exception("Error embedding with model '%s'", model_id)
            error_context = {"model_id": model_id}
            if alias_context:
                error_context.update(alias_context)
            raise ProviderAPIError(
                f"Error embedding with model {model_id}", context=error_context
            ) from exc
        finally:
            duration = time.perf_counter() - start_perf
            self._logger.debug(f"Model {model_id} embedding took {duration:.3f}s")

    def stream_model(
        self, model_id: str, prompt: str, **kwargs: Any
    ) -> Generator[str, None, ChatResponse]:
        """Stream a model response while recording usage and cost metrics."""

        start_wall = datetime.now(UTC)
        start_perf = time.perf_counter()
        requested_model_id = model_id
        model_id, alias_defaults = canonicalize_model_identifier(model_id)
        alias_context: dict[str, Any] = {}
        if model_id != requested_model_id:
            alias_context["requested_model_id"] = requested_model_id

        reservation = kwargs.pop("_reservation", None)
        if reservation is not None and not isinstance(reservation, Reservation):
            reservation = None

        kwargs.pop("stream", None)  # streaming handled explicitly

        for key, value in alias_defaults.items():
            if key == "reasoning" and "reasoning" in kwargs:
                continue
            if isinstance(value, dict):
                kwargs.setdefault(key, dict(value))
            else:
                kwargs.setdefault(key, value)

        provider_name, model_name = resolve_model_id(model_id)
        self._logger.debug(
            "stream resolve: %s -> provider=%s, model=%s", model_id, provider_name, model_name
        )

        if reservation is not None:
            requested_model = reservation.model or model_id
            if reservation.model:
                canonical_reservation_id, reservation_defaults = canonicalize_model_identifier(
                    requested_model
                )
                if canonical_reservation_id != requested_model:
                    alias_context["reservation_model_id"] = requested_model
                    requested_model = canonical_reservation_id
                for key, value in reservation_defaults.items():
                    if key == "reasoning" and "reasoning" in kwargs:
                        continue
                    if isinstance(value, dict):
                        kwargs.setdefault(key, dict(value))
                    else:
                        kwargs.setdefault(key, value)
            provider_override, model_override = resolve_model_id(requested_model)
            if provider_override != "unknown":
                provider_name = provider_override
            elif reservation.provider:
                provider_name = reservation.provider
            if model_override:
                model_name = model_override
            model_id = requested_model

        model = self.get_model(model_id)

        if "model_invocations" in self._metrics:
            self._metrics["model_invocations"].labels(model_id=model_id).inc()

        invocation_kwargs = dict(kwargs)
        if reservation is not None and reservation.credentials:
            cred_overrides = dict(reservation.credentials)
            cred_overrides.update(invocation_kwargs)
            invocation_kwargs = cred_overrides

        def _stream() -> Generator[str, None, ChatResponse]:
            try:
                stream = model.stream_complete(prompt, model_name, **invocation_kwargs)
                while True:
                    chunk = next(stream)
                    yield chunk
            except StopIteration as stop:
                response = stop.value
                if not isinstance(response, ChatResponse):
                    raise ProviderAPIError(
                        "Streaming provider must return ChatResponse",
                        context={"model_id": model_id},
                    ) from stop

                if not response.model_id:
                    response.model_id = model_id

                if response.started_at is None:
                    response.started_at = start_wall
                if response.completed_at is None:
                    response.completed_at = datetime.now(UTC)
                response.latency_ms = (time.perf_counter() - start_perf) * 1000.0

                if response.usage:
                    usage = response.usage
                    estimated_cost = self._calculate_cost(model_id, usage)
                    actual_cost = usage.actual_cost_usd

                    tracker_sample = UsageStats(
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        total_tokens=usage.total_tokens,
                        cost_usd=estimated_cost,
                        actual_cost_usd=actual_cost,
                    )
                    track_usage(tracker_sample, model_id)

                    usage.cost_usd = actual_cost if actual_cost is not None else estimated_cost
                    self._track_usage(model_id, usage)

                return response
            except ProviderAPIError:
                raise
            except Exception as e:
                self._logger.exception(f"Error streaming model '{model_id}'")
                error_context = {"model_id": model_id}
                if alias_context:
                    error_context.update(alias_context)
                raise ProviderAPIError(
                    f"Error streaming model {model_id}", context=error_context
                ) from e
            finally:
                duration = time.perf_counter() - start_perf
                self._logger.debug(f"Model {model_id} streaming took {duration:.3f}s")

        return _stream()

    async def invoke_model_async(self, model_id: str, prompt: str, **kwargs: Any) -> ChatResponse:
        """Invoke a model asynchronously with tracking semantics.

        Args:
            model_id: Model identifier.
            prompt: Prompt text forwarded to the provider.
            **kwargs: Provider-specific options.

        Returns:
            ChatResponse: Response with usage and cost fields populated.

        Raises:
            ProviderAPIError: If the provider call fails.

        Examples:
            >>> import asyncio
            >>> asyncio.run(registry.invoke_model_async('gpt-4', 'Hi'))  # doctest: +SKIP
        """
        start_wall = datetime.now(UTC)
        start_perf = time.perf_counter()
        requested_model_id = model_id
        model_id, alias_defaults = canonicalize_model_identifier(model_id)
        alias_context: dict[str, Any] = {}
        if model_id != requested_model_id:
            alias_context["requested_model_id"] = requested_model_id
        for key, value in alias_defaults.items():
            kwargs.setdefault(key, value)

        provider_name, model_name = resolve_model_id(model_id)
        self._logger.debug(
            "resolve: %s -> provider=%s, model=%s", model_id, provider_name, model_name
        )
        model = self.get_model(model_id)

        try:
            # Check if provider supports async natively
            from typing import cast

            if asyncio.iscoroutinefunction(model.complete):
                resp_any = await model.complete(prompt, model_name, **kwargs)
            else:
                # Run sync model in thread pool to avoid blocking
                resp_any = await asyncio.to_thread(model.complete, prompt, model_name, **kwargs)

            response = cast(ChatResponse, resp_any)

            # Calculate cost and track usage
            if response.usage:
                usage = response.usage
                estimated_cost = self._calculate_cost(model_id, usage)
                actual_cost = usage.actual_cost_usd

                tracker_sample = UsageStats(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    cost_usd=estimated_cost,
                    actual_cost_usd=actual_cost,
                )
                track_usage(tracker_sample, model_id)

                usage.cost_usd = actual_cost if actual_cost is not None else estimated_cost
                self._track_usage(model_id, usage)

            duration_ms = (time.perf_counter() - start_perf) * 1000.0
            response.latency_ms = duration_ms
            if response.started_at is None:
                response.started_at = start_wall
            if response.completed_at is None:
                response.completed_at = datetime.now(UTC)

            return response

        except (ProviderAPIError, ModelNotFoundError, ModelProviderError, PricingNotFoundError):
            raise
        except Exception as exc:
            self._logger.exception("Unexpected async error invoking model '%s'", model_id)
            error_context = {"model_id": model_id}
            if alias_context:
                error_context.update(alias_context)
            raise ProviderAPIError(
                f"Async error invoking model {model_id}",
                context=error_context,
            ) from exc

    def _calculate_cost(self, model_id: str, usage: UsageStats) -> float:
        """Compute the estimated USD cost for a usage record.

        Args:
            model_id: Identifier of the model being billed.
            usage: Usage statistics returned from the provider.

        Returns:
            float: Estimated cost rounded to six decimal places.
        """
        cost_info = get_model_cost(model_id)

        # Calculate input and output costs separately (per 1M tokens)
        input_cost = (usage.prompt_tokens / 1_000_000.0) * cost_info["input"]
        output_cost = (usage.completion_tokens / 1_000_000.0) * cost_info["output"]
        return round(input_cost + output_cost, 6)  # Round to 6 decimal places

    def _track_usage(self, model_id: str, usage: UsageStats) -> None:
        """Append a usage sample to the registry history.

        Args:
            model_id: Model identifier associated with the usage.
            usage: Usage statistics to record.
        """
        with self._lock:
            if model_id not in self._usage_records:
                self._usage_records[model_id] = []
            self._usage_records[model_id].append(usage)

            # Limit history to last 1000 records per model
            if len(self._usage_records[model_id]) > 1000:
                self._usage_records[model_id] = self._usage_records[model_id][-1000:]

    def get_usage_summary(self, model_id: str) -> Optional[UsageStats]:
        """Return aggregated usage statistics for the given model.

        Args:
            model_id: Model identifier whose usage should be summarized.

        Returns:
            UsageStats | None: Combined usage metrics if records exist, otherwise ``None``.

        Examples:
            >>> registry.get_usage_summary('gpt-4')  # doctest: +SKIP
        """
        with self._lock:
            records = self._usage_records.get(model_id, [])
            if not records:
                return None

            # Aggregate all usage records
            total = UsageStats()
            for record in records:
                total.add(record)
            return total

    @staticmethod
    def _resolve_context() -> Optional["EmberContext"]:
        try:
            from ember._internal.context.runtime import EmberContext as _EmberContext
        except ImportError as exc:
            raise RuntimeError(
                "ember._internal.context.runtime is required for implicit context resolution"
            ) from exc

        return _EmberContext.current()

    def list_models(self) -> list[str]:
        """Return model identifiers currently cached by the registry.

        Returns:
            list[str]: Model identifiers for instantiated providers.

        Examples:
            >>> registry.list_models()
            []
        """
        return list(self._models.keys())

    def clear_cache(self) -> None:
        """Remove all cached provider instances.

        Examples:
            >>> registry.clear_cache()
        """
        with self._lock:
            self._models.clear()
            self._logger.info("Model cache cleared")
