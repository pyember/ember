"""Vertex AI provider for Gemini models hosted on Vertex."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from ember._internal.exceptions import ProviderAPIError
from ember.models.providers.base import BaseProvider
from ember.models.schemas import ChatResponse, UsageStats


class VertexAIProvider(BaseProvider):
    """Vertex AI provider using the Vertex Generative AI SDK."""

    requires_api_key = False

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ):
        """Initialise the Vertex AI provider."""
        super().__init__(api_key)
        self._vertexai = _import_vertex_ai()

        self._project = project or _config_str("project")
        if not self._project:
            raise ValueError("providers.vertex.project not configured")

        self._location = location or _config_str("location") or "us-central1"
        self._vertexai.init(project=self._project, location=self._location)

        try:
            from vertexai.generative_models import GenerativeModel
        except ImportError as exc:  # pragma: no cover
            raise ProviderAPIError(
                "vertexai.generative_models module not available",
                context={"provider": "vertex"},
            ) from exc

        self._model_cls = GenerativeModel

    def complete(self, prompt: str, model: str, **kwargs: Any) -> ChatResponse:
        """Generate content using Vertex AI."""
        stream_requested = kwargs.pop("stream", False)
        if stream_requested:
            raise ProviderAPIError(
                "Streaming responses are not implemented for Vertex AI",
                context={"model": model, "error_type": "not_implemented"},
            )

        system = kwargs.pop("system", None)
        context = kwargs.pop("context", None)
        input_text = _compose_prompt(prompt, system, context)

        generation_config = _build_generation_config(kwargs)
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ProviderAPIError(
                f"Unsupported parameters for Vertex AI: {unsupported}",
                context={"model": model, "unsupported": tuple(sorted(kwargs))},
            )

        try:
            model_client = self._model_cls(model)
        except Exception as exc:
            raise ProviderAPIError(
                f"Failed to initialise Vertex model '{model}': {exc}",
                context={"model": model},
            ) from exc

        try:
            response = model_client.generate_content(  # type: ignore[no-untyped-call]
                input_text,
                generation_config=(generation_config if generation_config else None),
            )
        except Exception as exc:
            raise ProviderAPIError(
                f"Vertex AI request failed: {exc}",
                context={"model": model},
            ) from exc

        text = _extract_text(response)
        usage = _usage_from_vertex(response)
        return ChatResponse(
            data=text,
            usage=usage,
            model_id=model,
            raw_output=response,
        )

    def _get_api_key_from_env(self) -> str:
        return ""


def _import_vertex_ai():
    try:
        import vertexai  # type: ignore
    except ImportError as exc:
        raise ProviderAPIError(
            "Vertex AI SDK not installed. Install with `pip install google-cloud-aiplatform`.",
            context={"provider": "vertex"},
        ) from exc
    return vertexai


def _config_value(field: str) -> Any:
    from ember._internal.context.runtime import EmberContext

    return EmberContext.current().get_config(f"providers.vertex.{field}")


def _config_str(field: str) -> Optional[str]:
    raw = _config_value(field)
    if raw is None:
        return None
    if isinstance(raw, str):
        trimmed = raw.strip()
        return trimmed or None
    raise TypeError(f"providers.vertex.{field} must be a string")


def _compose_prompt(
    prompt: str,
    system: Optional[str],
    context: Optional[Any],
) -> str:
    parts: list[str] = []
    if system:
        parts.append(str(system))
    if context is not None:
        if isinstance(context, str):
            parts.append(context)
        elif isinstance(context, Sequence) and not isinstance(context, (bytes, bytearray, str)):
            parts.extend(str(entry) for entry in context)
        else:
            raise TypeError("Vertex context must be a string or sequence of strings")
    parts.append(prompt)
    return "\n\n".join(filter(None, parts))


def _build_generation_config(kwargs: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if "temperature" in kwargs:
        config["temperature"] = kwargs.pop("temperature")
    if "max_tokens" in kwargs:
        config["max_output_tokens"] = kwargs.pop("max_tokens")
    if "top_p" in kwargs:
        config["top_p"] = kwargs.pop("top_p")
    if "stop" in kwargs:
        config["stop_sequences"] = kwargs.pop("stop")
    return config


def _extract_text(response: Any) -> str:
    if response is None:
        return ""
    if getattr(response, "text", None):
        return str(response.text)

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""

    fragments: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                fragments.append(str(text))
    return "\n".join(fragments)


def _usage_from_vertex(response: Any) -> Optional[UsageStats]:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return None
    try:
        prompt_tokens = int(getattr(metadata, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(metadata, "candidates_token_count", 0) or 0)
        total_tokens = int(
            getattr(metadata, "total_token_count", prompt_tokens + completion_tokens)
        )
    except (TypeError, ValueError):
        prompt_tokens = completion_tokens = total_tokens = 0

    return UsageStats(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
