"""Anthropic provider for Claude models."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from ember._internal.exceptions import (
    ProviderAPIError,
)
from ember.models.discovery.provider_api import DiscoveryProvider
from ember.models.discovery.registry import register_provider as register_discovery_provider
from ember.models.discovery.types import DiscoveredModel
from ember.models.providers.base import BaseProvider
from ember.models.providers.utils import (
    ResponsesRequest,
    build_prompt_components,
    parse_responses_request,
    text_turns,
)
from ember.models.schemas import ChatResponse, UsageStats

ContextMapping = Mapping[str, object]
SystemBlock = Mapping[str, str]
AnthropicMessage = Mapping[str, object]


def _normalize_context(
    raw: Sequence[ContextMapping] | ContextMapping | None,
) -> tuple[ContextMapping, ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        return (raw,)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(raw)
    raise ProviderAPIError(
        "Anthropic context must be a mapping or sequence of mappings",
        context={"type": type(raw).__name__},
    )


def _anthropic_messages(prompt: str, context: Sequence[ContextMapping]) -> list[AnthropicMessage]:
    prompt_text = prompt.strip()
    if not prompt_text:
        raise ProviderAPIError("Anthropic prompt must be non-empty")

    messages: list[AnthropicMessage] = []
    for index, entry in enumerate(context):
        try:
            role_raw = entry["role"]
        except KeyError as exc:
            raise ProviderAPIError(
                "Anthropic context entries must include a role",
                context={"index": index},
            ) from exc
        role = str(role_raw).strip()
        if not role:
            raise ProviderAPIError(
                "Anthropic context roles must be non-empty",
                context={"index": index},
            )
        content_block = _text_block(entry.get("content"), field=f"context[{index}]")
        messages.append({"role": role, "content": [content_block]})

    messages.append({"role": "user", "content": [_text_block(prompt_text, field="prompt")]})
    return messages


def _normalize_system(
    raw: Sequence[str | Mapping[str, str]] | str | Mapping[str, str] | None,
) -> list[SystemBlock] | None:
    if raw is None:
        return None
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        blocks = [_text_block(part, field=f"system[{index}]") for index, part in enumerate(raw)]
        if not blocks:
            raise ProviderAPIError("Anthropic system prompt sequences must contain text")
        return blocks
    return [_text_block(raw, field="system")]


def _text_block(raw: str | Mapping[str, str], *, field: str) -> SystemBlock:
    if isinstance(raw, Mapping):
        text = str(raw.get("text", "")).strip()
        block_type = str(raw.get("type", "text") or "text")
    else:
        text = str(raw).strip()
        block_type = "text"
    if not text:
        raise ProviderAPIError(
            "Anthropic messages require non-empty text",
            context={"field": field},
        )
    return {"type": block_type, "text": text}


logger = logging.getLogger(__name__)


def _resolve_anthropic_api_key() -> str:
    """Resolve Anthropic API key from the active Ember context."""
    from ember._internal.context.runtime import EmberContext

    return EmberContext.current().get_credential("anthropic")


class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider with retry-aware completions.

    Handles API invocation, retries, and normalization into ``ChatResponse``.
    """

    supports_responses_api = True

    def __init__(self, api_key: str | None = None):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key. When omitted, resolves from configuration.
        """
        super().__init__(api_key)
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _resolve_model_name(self, model: str) -> str:
        """Resolve user-friendly model names to actual API model IDs.

        Maps short names like 'claude-3-haiku' to full API names like
        'claude-3-haiku-20240307'. This provides a better user experience
        while maintaining API compatibility.

        Args:
            model: User-provided model name.

        Returns:
            Resolved model name for API calls.
        """
        # Model name mappings from friendly names to API names. This deliberately
        # enumerates only the aliases that Anthropic documents publicly so that
        # resolution stays explicit.
        return _MODEL_NAME_ALIASES.get(model, model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True,
    )
    def complete(self, prompt: str, model: str, **kwargs: Any) -> ChatResponse:
        """Complete a prompt using Anthropic's API.

        Args:
            prompt: The prompt text.
            model: Model name (e.g., "claude-3-opus", "claude-3-sonnet").
            **kwargs: Additional parameters like temperature, max_tokens, etc.

        Returns:
            ChatResponse with completion and usage information.

        Raises:
            ProviderAPIError: For API errors.
            AuthenticationError: For invalid API key.
            RateLimitError: When rate limited.
        """
        resolved_model = self._resolve_model_name(model)

        params = self._build_completion_params(
            prompt=prompt,
            requested_model=model,
            resolved_model=resolved_model,
            raw_kwargs=kwargs,
        )

        message_count = len(params.get("messages", ()))

        try:
            # Make API call
            logger.debug(
                "Anthropic API call: model=%s (requested: %s), messages=%d",
                resolved_model,
                model,
                message_count,
            )
            # Avoid tight coupling to SDK stubs/overloads
            response = self.client.messages.create(**params)

            return self._build_chat_response(response, model)

        except ValueError as exc:
            if _STREAMING_RECOMMENDATION in str(exc):
                logger.info(
                    "Anthropic requested streaming for model %s; retrying via stream",
                    resolved_model,
                )
                return self._complete_with_stream(params, model, resolved_model)
            raise

        except anthropic.AuthenticationError as e:
            logger.error(f"Anthropic authentication error: {e}")
            raise ProviderAPIError(
                "Invalid Anthropic API key",
                context={
                    "model": model,
                    "resolved_model": resolved_model,
                    "error_type": "authentication",
                },
            ) from e

        except anthropic.RateLimitError as e:
            logger.warning(f"Anthropic rate limit: {e}")
            raise ProviderAPIError(
                "Anthropic rate limit exceeded",
                context={
                    "model": model,
                    "resolved_model": resolved_model,
                    "error_type": "rate_limit",
                },
            ) from e

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise ProviderAPIError(
                f"Anthropic API error: {str(e)}",
                context={"model": model, "resolved_model": resolved_model},
            ) from e

        except Exception as e:
            logger.exception("Unexpected error calling Anthropic API")
            raise ProviderAPIError(
                f"Unexpected error: {str(e)}",
                context={"model": model, "resolved_model": resolved_model},
            ) from e

    def complete_responses_payload(
        self,
        payload: Mapping[str, Any],
        **overrides: Any,
    ) -> ChatResponse:
        request = self._parse_responses_payload(payload)
        if overrides:
            unsupported = ", ".join(sorted(overrides))
            raise ProviderAPIError(
                f"Unsupported Anthropic Responses overrides: {unsupported}",
                context={"model": request.model},
            )

        self._validate_responses_payload(request)
        turns = text_turns(request)
        if not turns:
            raise ProviderAPIError(
                "Responses payload must include at least one user message",
                context={"model": request.model},
            )
        if any(role.lower() not in {"user", "assistant"} for role, _ in turns):
            raise ProviderAPIError(
                "Anthropic Responses adapter only supports user/assistant roles",
                context={"model": request.model},
            )
        if not any(role.lower() == "user" for role, _ in turns):
            raise ProviderAPIError(
                "Anthropic Responses payload requires a user turn",
                context={"model": request.model},
            )

        components = build_prompt_components(request)
        params: dict[str, Any] = {}
        if request.max_output_tokens is not None:
            params["max_tokens"] = request.max_output_tokens
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.stop_sequences:
            params["stop"] = list(request.stop_sequences)

        # Pass tools and tool_choice to native Anthropic API
        if request.tools:
            params["tools"] = list(request.tools)
        if request.tool_choice is not None:
            params["tool_choice"] = request.tool_choice

        context_messages = [dict(item) for item in components.context]

        return self.complete(
            components.prompt,
            request.model,
            system=components.system,
            context=context_messages,
            **params,
        )

    def stream_responses_payload(
        self,
        payload: Mapping[str, Any],
        **_: Any,
    ) -> Iterable[str]:
        raise ProviderAPIError(
            "Streaming is not implemented for Anthropic Responses payloads",
            context={"model": payload.get("model")},
        )

    def _build_chat_response(self, response: Any, requested_model: str) -> ChatResponse:
        """Build ChatResponse from Anthropic API response."""
        content = getattr(response, "content", ())

        # Separate text and thinking blocks
        text_parts: list[str] = []
        thinking_parts: list[str] = []

        for chunk in content:
            chunk_type = getattr(chunk, "type", None)
            if chunk_type is None and isinstance(chunk, Mapping):
                chunk_type = chunk.get("type")

            if chunk_type == "thinking":
                # Extract thinking trace from thinking blocks
                thinking = getattr(chunk, "thinking", None)
                if thinking is None and isinstance(chunk, Mapping):
                    thinking = chunk.get("thinking")
                if thinking:
                    thinking_parts.append(str(thinking))
            else:
                # Extract text from text blocks (default behavior)
                text_parts.append(_content_piece_text(chunk))

        text = "".join(text_parts)
        thinking_trace = "\n".join(thinking_parts) if thinking_parts else None

        usage_payload = getattr(response, "usage", None)
        usage = None
        if usage_payload is not None:
            usage = UsageStats(
                prompt_tokens=usage_payload.input_tokens,
                completion_tokens=usage_payload.output_tokens,
                total_tokens=usage_payload.input_tokens + usage_payload.output_tokens,
            )

        return ChatResponse(
            data=text,
            usage=usage,
            model_id=requested_model,
            raw_output=response,
            thinking_trace=thinking_trace,
        )

    def _complete_with_stream(
        self,
        params: Mapping[str, Any],
        requested_model: str,
        resolved_model: str,
    ) -> ChatResponse:
        stream_params = dict(params)
        try:
            with self.client.messages.stream(**stream_params) as stream:
                for _ in stream:
                    continue
                final_response = stream.get_final_response()
        except Exception as exc:
            logger.exception("Anthropic streaming fallback failed")
            raise ProviderAPIError(
                "Anthropic streaming fallback failed",
                context={"model": requested_model, "resolved_model": resolved_model},
            ) from exc

        return self._build_chat_response(final_response, requested_model)

    def _build_completion_params(
        self,
        *,
        prompt: str,
        requested_model: str,
        resolved_model: str,
        raw_kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Assemble Anthropic SDK arguments from high-level options."""

        options = dict(raw_kwargs)

        context_entries = _normalize_context(options.pop("context", None))
        system_blocks = _normalize_system(options.pop("system", None))
        messages = _anthropic_messages(prompt, context_entries)

        # Extract thinking config to auto-adjust max_tokens if needed
        thinking_config = options.get("thinking")
        thinking_budget: int | None = None
        if isinstance(thinking_config, Mapping):
            budget_value = thinking_config.get("budget_tokens")
            if budget_value is not None:
                try:
                    thinking_budget = int(budget_value)
                except (TypeError, ValueError):
                    pass

        max_tokens = _resolve_max_tokens(
            explicit=options.pop("max_tokens", None),
            override=options.pop("max_output_tokens", None),
            model=requested_model,
            resolved_model=resolved_model,
            thinking_budget=thinking_budget,
        )

        params: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system_blocks:
            params["system"] = system_blocks

        passthrough_keys = {
            "temperature": "temperature",
            "top_p": "top_p",
            "stop": "stop_sequences",
        }
        for source, target in passthrough_keys.items():
            if source in options:
                params[target] = options.pop(source)

        params.update(options)
        return params

    @staticmethod
    def _parse_responses_payload(payload: Mapping[str, Any]) -> ResponsesRequest:
        if not isinstance(payload, Mapping):
            raise ProviderAPIError("Responses payload must be a JSON object")
        try:
            return parse_responses_request(dict(payload))
        except ValueError as exc:
            raise ProviderAPIError(str(exc)) from exc

    @staticmethod
    def _validate_responses_payload(request: ResponsesRequest) -> None:
        # Note: tools and tool_choice are now supported via native Anthropic API
        if request.response_format:
            raise ProviderAPIError(
                "Anthropic Responses adapter does not support response_format",
                context={"model": request.model},
            )
        if request.reasoning is not None:
            raise ProviderAPIError(
                "Anthropic Responses adapter does not support reasoning controls",
                context={"model": request.model},
            )

    def _get_api_key_from_env(self) -> str:
        """Get Anthropic API key from configuration."""

        return _resolve_anthropic_api_key()

    def validate_model(self, model: str) -> bool:
        """Check if this provider supports the given model.

        Args:
            model: Model name to validate.

        Returns:
            True if model is supported.
        """
        from ember.models.catalog import list_available_models

        available = set(list_available_models(provider="anthropic"))
        if model in available:
            return True

        resolved = self._resolve_model_name(model)
        return resolved in available

    def get_model_info(self, model: str) -> dict[str, Any]:
        """Get information about a specific model.

        Args:
            model: Model name.

        Returns:
            Dictionary with model information.
        """
        from ember.models.catalog import get_model_info

        info = super().get_model_info(model)

        resolved_model = self._resolve_model_name(model)
        lookup_order = (resolved_model, model)
        for candidate in lookup_order:
            try:
                catalog_entry = get_model_info(candidate)
            except KeyError:
                continue
            info.update(
                {
                    "context_window": catalog_entry.context_window,
                    "status": getattr(catalog_entry, "status", "stable"),
                    "description": catalog_entry.description,
                }
            )
            break

        return info


class AnthropicDiscoveryAdapter(DiscoveryProvider):
    """Discovery adapter leveraging Anthropic's model listing endpoint."""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key_resolver=_resolve_anthropic_api_key,
        client_factory=None,
    ) -> None:
        self._resolve_api_key = api_key_resolver
        if client_factory is None:

            def _default_factory(key: str) -> anthropic.Anthropic:
                return anthropic.Anthropic(api_key=key)

            client_factory = _default_factory
        self._client_factory = client_factory

    def list_models(
        self,
        *,
        region: str | None = None,
        project_hint: str | None = None,
    ) -> Iterable[DiscoveredModel]:
        """List available models from Anthropic API.

        Args:
            region: Region hint (ignored for Anthropic).
            project_hint: Project hint (ignored for Anthropic).

        Returns:
            List of discovered models.
        """
        api_key = self._resolve_api_key()

        client = self._client_factory(api_key)

        if region:
            logger.debug("Anthropic discovery region hint '%s' ignored", region)
        if project_hint:
            logger.debug("Anthropic discovery project hint '%s' ignored", project_hint)

        paginator = client.models.list(limit=_DISCOVERY_PAGE_LIMIT)  # type: ignore[no-untyped-call]
        iterator = getattr(paginator, "iter_pages", None)
        pages = iterator() if callable(iterator) else (paginator,)

        models: list[DiscoveredModel] = []
        for page in pages:
            entries = getattr(page, "data", None) or []
            for entry in entries:
                discovered = self._to_discovered(entry)
                if discovered is not None:
                    models.append(discovered)
        return models

    def _to_discovered(self, entry: Any) -> DiscoveredModel | None:
        """Convert API entry to DiscoveredModel."""
        model_id = getattr(entry, "id", None)
        if model_id is None and isinstance(entry, dict):
            model_id = entry.get("id")
        if not model_id:
            return None

        display = getattr(entry, "display_name", None) or getattr(entry, "displayName", None)
        description = getattr(entry, "description", None)
        raw_payload = _extract_payload(entry)

        return DiscoveredModel(
            provider="anthropic",
            id=model_id,
            display_name=display,
            description=description,
            raw_payload=raw_payload,
        )


def _extract_payload(entry: Any) -> dict[str, Any]:
    """Extract dictionary payload from an API entry object.

    Args:
        entry: API response entry object.

    Returns:
        Dictionary representation of the entry.
    """
    if isinstance(entry, dict):
        return dict(entry)

    # Try standard serialization methods
    if hasattr(entry, "model_dump") and callable(entry.model_dump):
        return entry.model_dump()

    if hasattr(entry, "to_dict") and callable(entry.to_dict):
        return entry.to_dict()

    # If no serialization method found, return empty dict
    return {}


register_discovery_provider(AnthropicDiscoveryAdapter())
_STREAMING_RECOMMENDATION = "Streaming is strongly recommended"
_MODEL_NAME_ALIASES: Final[dict[str, str]] = {
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
    "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-4-opus": "claude-opus-4-20250514",
    "claude-opus-4.1": "claude-opus-4-1-20250805",
    "claude-4.1-opus": "claude-opus-4-1-20250805",
    "claude-4-sonnet": "claude-4-sonnet-20250514",
    "claude-4.5-sonnet": "claude-sonnet-4-5-20250929",
    "claude-4-5-sonnet": "claude-sonnet-4-5-20250929",
    "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
    "claude-sonnet-4": "claude-4-sonnet-20250514",
    # Legacy models
    "claude-2.1": "claude-2.1",
    "claude-2.0": "claude-2.0",
    "claude-instant-1.2": "claude-instant-1.2",
}
_DEFAULT_MAX_TOKENS: Final[int] = 4096
_DISCOVERY_PAGE_LIMIT: Final[int] = 100


def _content_piece_text(chunk: Any) -> str:
    if hasattr(chunk, "text"):
        return str(chunk.text)
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, Mapping):
        text = chunk.get("text")
        return str(text) if text is not None else ""
    return ""


def _resolve_max_tokens(
    *,
    explicit: Any,
    override: Any,
    model: str,
    resolved_model: str,
    thinking_budget: int | None = None,
) -> int:
    if explicit is not None and override is not None:
        raise ProviderAPIError(
            "Specify only one of 'max_tokens' or 'max_output_tokens' for Anthropic requests",
            context={"model": model, "resolved_model": resolved_model},
        )

    raw_value = override if override is not None else explicit
    value = raw_value if raw_value is not None else _DEFAULT_MAX_TOKENS

    try:
        max_tokens = int(value)
    except (TypeError, ValueError) as exc:
        raise ProviderAPIError(
            "Anthropic 'max_tokens' must be an integer",
            context={"model": model, "resolved_model": resolved_model},
        ) from exc

    if max_tokens <= 0:
        raise ProviderAPIError(
            "Anthropic 'max_tokens' must be positive",
            context={"model": model, "resolved_model": resolved_model},
        )

    # Auto-adjust max_tokens when thinking is enabled
    # Anthropic requires max_tokens > budget_tokens for extended thinking
    if thinking_budget is not None and max_tokens <= thinking_budget:
        max_tokens = thinking_budget + _DEFAULT_MAX_TOKENS
        logger.debug(
            "Auto-adjusted max_tokens to %d (thinking budget: %d)",
            max_tokens,
            thinking_budget,
        )

    return max_tokens
