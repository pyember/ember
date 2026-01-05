"""Google Generative AI provider for Gemini models.

Examples:
    >>> from ember.models.providers.google import GoogleProvider
"""

import logging
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from ember._internal.exceptions import (
    ModelProviderError,
    ProviderAPIError,
)
from ember.models.discovery.provider_api import DiscoveryProvider
from ember.models.discovery.registry import register_provider as register_discovery_provider
from ember.models.discovery.types import DiscoveredModel
from ember.models.providers.base import BaseProvider
from ember.models.providers.utils import (
    PromptComponents,
    ResponsesRequest,
    build_prompt_components,
    parse_responses_request,
    text_turns,
)
from ember.models.schemas import ChatResponse, EmbeddingResponse, UsageStats

genai: Any = None


_MODEL_ALIASES: Dict[str, str] = {
    # Legacy Gemini 1.0 identifiers → Gemini 1.5 latest endpoints
    "models/gemini-pro": "models/gemini-1.5-pro-latest",
    "models/gemini-pro-vision": "models/gemini-1.5-pro-latest",
    "models/gemini-1.5-pro": "models/gemini-1.5-pro-latest",
    # Ensure bare aliases without "-latest" resolve to the newest release
    "models/gemini-1.5-flash": "models/gemini-1.5-flash-latest",
}

_MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    "models/gemini-1.5-pro-latest": 2_000_000,
    "models/gemini-1.5-flash-latest": 1_000_000,
}


def _load_genai_module() -> Any:
    global genai
    if genai is not None:
        return genai

    try:
        from google import genai as genai_import  # type: ignore
    except ImportError:
        try:
            import google.generativeai as genai_import  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ModelProviderError(
                "Google Gen AI library not installed. Install with: pip install google-genai"
            ) from exc

    globals()["genai"] = genai_import  # type: ignore[assignment]
    return genai_import


logger = logging.getLogger(__name__)


def _resolve_google_api_key() -> str:
    """Resolve Google API key from the active Ember context."""
    from ember._internal.context.runtime import EmberContext

    return EmberContext.current().get_credential("google")


class GoogleProvider(BaseProvider):
    """Google Generative AI provider for Gemini chat models."""

    supports_responses_api = True

    def __init__(self, api_key: str | None = None):
        """Initialize Google provider.

        Args:
            api_key: Google API key override.
        """
        super().__init__(api_key)
        self._apply_pyasn1_patch()
        self._ensure_imports()
        genai.configure(api_key=self.api_key)  # type: ignore[attr-defined]

    def _apply_pyasn1_patch(self) -> None:
        """Apply a pyasn1 compatibility patch.

        This defers the import and avoids issues during static type checking.

        Returns:
            None
        """
        try:
            import importlib

            mod = importlib.import_module("ember._internal.patches.pyasn1")
            func = getattr(mod, "ensure_pyasn1_compatibility", None)
            if callable(func):
                func()
        except Exception as exc:
            logger.warning("pyasn1 compatibility patch failed", exc_info=exc)

    def _ensure_imports(self) -> None:
        """Ensure Google Generative AI imports are available.

        Uses lazy loading to avoid protobuf issues at module import time.

        Returns:
            None
        """
        global genai
        if genai is not None:
            return

        try:
            module = _load_genai_module()
        except TypeError as e:
            if "got multiple values for keyword argument '_options'" in str(e):
                raise ModelProviderError(
                    "Google Gen AI import failed due to protobuf version conflict. "
                    "Install protobuf<3.20 or pin google-genai to a compatible version."
                ) from e
            raise

        globals()["genai"] = module  # type: ignore[assignment]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True,
    )
    def complete(self, prompt: str, model: str, **kwargs: Any) -> ChatResponse:
        """Complete a prompt using Google's API.

        Args:
            prompt: The prompt text.
            model: Model name (e.g., "gemini-1.5-pro-latest", "gemini-1.5-flash-latest").
            **kwargs: Additional parameters like temperature, max_tokens, etc.

        Returns:
            ChatResponse with completion and usage information.

        Raises:
            ProviderAPIError: For API errors.
            AuthenticationError: For invalid API key.
        """
        requested_model = model

        try:
            model = self._normalize_model_name(model)
        except ModelProviderError as exc:
            logger.error("Invalid Google model '%s': %s", requested_model, exc)
            raise ProviderAPIError(
                f"Unsupported Google model: {requested_model}",
                context={"model": requested_model},
            ) from exc

        # Get the model
        try:
            gemini_model = genai.GenerativeModel(model)  # type: ignore[attr-defined]
        except Exception as e:
            logger.error(f"Failed to create Gemini model: {e}")
            raise ProviderAPIError(
                f"Failed to create Gemini model: {str(e)}", context={"model": model}
            ) from e

        # Build generation config
        generation_config = {}

        # Map common parameters
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            generation_config["max_output_tokens"] = kwargs.pop("max_tokens")
        if "top_p" in kwargs:
            generation_config["top_p"] = kwargs.pop("top_p")
        if "stop" in kwargs:
            generation_config["stop_sequences"] = kwargs.pop("stop")

        # Handle context/system message
        context = kwargs.pop("context", None)
        if context:
            prompt = f"{context}\n\n{prompt}"

        try:
            # Make API call
            logger.debug(f"Google API call: model={model}")
            response = gemini_model.generate_content(  # type: ignore[no-untyped-call]
                prompt,
                generation_config=(generation_config or None),
            )

            text = self._coerce_response_text(response)

            if not text:
                finish_reasons = [
                    getattr(candidate, "finish_reason", None)
                    for candidate in getattr(response, "candidates", []) or []
                ]
                logger.debug(
                    "Google response contained no text; finish_reasons=%s",
                    finish_reasons,
                )

            # Build usage stats
            usage = None
            try:
                um = getattr(response, "usage_metadata", None)
                if um is not None:
                    # Prefer SDK-provided metadata when available
                    pt = int(getattr(um, "prompt_token_count", 0) or 0)
                    ct = int(getattr(um, "candidates_token_count", 0) or 0)
                    tt = int(getattr(um, "total_token_count", 0) or (pt + ct))
                    usage = UsageStats(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)
            except Exception:
                # Fall back to estimate below
                usage = None

            if usage is None:
                # Estimate when metadata is unavailable
                pt_est = len(prompt.split()) * 2
                ct_est = len(text.split()) * 2
                usage = UsageStats(
                    prompt_tokens=pt_est,
                    completion_tokens=ct_est,
                    total_tokens=pt_est + ct_est,
                )

            return ChatResponse(
                data=text,
                usage=usage,
                model_id=model,
                raw_output=response,
            )

        except Exception as e:
            error_str = str(e)

            if "API_KEY_INVALID" in error_str or "invalid api key" in error_str.lower():
                logger.error(f"Google authentication error: {e}")
                raise ProviderAPIError(
                    "Invalid Google API key",
                    context={"model": model, "error_type": "authentication"},
                ) from e

            elif "RATE_LIMIT_EXCEEDED" in error_str:
                logger.warning(f"Google rate limit: {e}")
                raise ProviderAPIError(
                    "Google rate limit exceeded",
                    context={"model": model, "error_type": "rate_limit"},
                ) from e

            else:
                logger.error(f"Google API error: {e}")
            raise ProviderAPIError(
                f"Google API error: {error_str}", context={"model": model}
            ) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def embed(
        self,
        inputs: Sequence[str],
        model: str,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Generate embeddings using Google's embedding models.

        Args:
            inputs: Sequence of text strings to embed.
            model: Embedding model name (e.g., "gemini-embedding-001").
            **kwargs: Additional parameters like dimensions.

        Returns:
            EmbeddingResponse with embedding vectors and usage info.
        """
        dimensions = kwargs.pop("dimensions", None)

        # Normalize model name - embedding models use different naming
        if not model.startswith("models/"):
            model = f"models/{model}"

        try:
            # Google API supports batch embedding - pass list directly
            # Max ~100 items per request, so chunk if needed
            chunk_size = 100
            embeddings = []
            total_tokens = 0

            for i in range(0, len(inputs), chunk_size):
                chunk = list(inputs[i : i + chunk_size])
                result = genai.embed_content(  # type: ignore[attr-defined]
                    model=model,
                    content=chunk,
                    output_dimensionality=dimensions,
                )

                # Extract embeddings from batch result
                chunk_embeddings = result.get("embedding", [])
                if not chunk_embeddings and hasattr(result, "embedding"):
                    chunk_embeddings = result.embedding

                # Handle both single and batch responses
                if chunk_embeddings and not isinstance(chunk_embeddings[0], list):
                    # Single embedding returned as flat list
                    embeddings.append(chunk_embeddings)
                else:
                    embeddings.extend(chunk_embeddings)

                # Estimate tokens
                total_tokens += sum(len(t.split()) * 2 for t in chunk)

            usage = UsageStats(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens,
            )

            return EmbeddingResponse(
                embeddings=embeddings,
                usage=usage,
                model_id=model,
                raw_output=None,
            )

        except Exception as e:
            error_str = str(e)
            if "API_KEY_INVALID" in error_str or "invalid api key" in error_str.lower():
                raise ProviderAPIError(
                    "Invalid Google API key",
                    context={"model": model, "error_type": "authentication"},
                ) from e
            raise ProviderAPIError(
                f"Google embedding error: {error_str}",
                context={"model": model},
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
                f"Unsupported Gemini Responses overrides: {unsupported}",
                context={"model": request.model},
            )

        self._validate_responses_payload(request)
        turns = text_turns(request)
        if not turns:
            raise ProviderAPIError(
                "Responses payload must include at least one user message",
                context={"model": request.model},
            )

        components = build_prompt_components(request)
        params: Dict[str, Any] = {}
        if request.max_output_tokens is not None:
            params["max_tokens"] = request.max_output_tokens
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.stop_sequences:
            params["stop"] = list(request.stop_sequences)

        context_blob = self._compose_context_text(components)

        return self.complete(
            components.prompt,
            request.model,
            context=context_blob,
            **params,
        )

    def stream_responses_payload(
        self,
        payload: Mapping[str, Any],
        **_: Any,
    ) -> Iterable[str]:
        raise ProviderAPIError(
            "Streaming is not implemented for Gemini Responses payloads",
            context={"model": payload.get("model")},
        )

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
        if request.tools:
            raise ProviderAPIError(
                "Gemini Responses adapter does not yet support tools",
                context={"model": request.model},
            )
        if request.response_format:
            raise ProviderAPIError(
                "Gemini Responses adapter does not yet support response_format",
                context={"model": request.model},
            )
        if request.reasoning is not None:
            raise ProviderAPIError(
                "Gemini Responses adapter does not support reasoning controls",
                context={"model": request.model},
            )
        if request.tool_choice is not None:
            raise ProviderAPIError(
                "Gemini Responses adapter does not yet support tool_choice",
                context={"model": request.model},
            )

    @staticmethod
    def _compose_context_text(components: PromptComponents) -> Optional[str]:
        segments: list[str] = []
        if components.system:
            segments.append(components.system)
        for entry in components.context:
            role = entry.get("role", "context")
            content = entry.get("content", "")
            if content:
                segments.append(f"{role}: {content}")

        if not segments:
            return None
        return "\n\n".join(segment.strip() for segment in segments if segment)

    def _coerce_response_text(self, response: Any) -> str:
        """Return best-effort text from a Gemini response.

        The Google SDK raises when ``response.text`` is accessed for candidates that
        terminate with ``MAX_TOKENS`` or safety blocks. We prefer the quick accessor
        when available and otherwise fold over the candidates manually.
        """

        try:
            text_value = response.text
        except Exception as text_error:  # pragma: no cover
            logger.debug(
                "Google response text accessor failed; falling back to candidate parts",
                exc_info=text_error,
            )
        else:
            if isinstance(text_value, str) and text_value:
                return text_value

        return self._extract_text_from_candidates(response)

    @staticmethod
    def _extract_text_from_candidates(response: Any) -> str:
        """Safely materialize text from candidate parts when quick accessor fails."""

        candidates = getattr(response, "candidates", None) or []
        collected: list[str] = []

        for candidate in candidates:
            parts = GoogleProvider._coerce_candidate_parts(candidate)
            if not parts:
                continue

            for part in parts:
                text = GoogleProvider._coerce_part_text(part)
                if text:
                    collected.append(text)

        return "\n".join(segment.strip() for segment in collected if segment).strip()

    @staticmethod
    def _coerce_candidate_parts(candidate: Any) -> Iterable[Any]:
        """Return the most likely ``parts`` iterable from a candidate."""

        content = getattr(candidate, "content", None)
        parts = None
        if content is not None:
            parts = getattr(content, "parts", None)
            if parts is None and isinstance(content, dict):
                parts = content.get("parts")

        if parts is None and hasattr(candidate, "parts"):
            parts = getattr(candidate, "parts", None)

        if not parts:
            return ()

        if isinstance(parts, (list, tuple)):
            return parts

        if isinstance(parts, Iterable) and not isinstance(parts, (str, bytes)):
            return parts

        return ()

    @staticmethod
    def _coerce_part_text(part: Any) -> Optional[str]:
        """Extract the ``text`` field from a part-like object."""

        if isinstance(part, str):
            return part

        text = getattr(part, "text", None)
        if text is None and isinstance(part, dict):
            text = part.get("text")

        return text if text else None

    def _get_api_key_from_env(self) -> str:
        """Resolve Google API key from the active Ember configuration."""
        from ember._internal.context.runtime import EmberContext

        return EmberContext.current().get_credential("google")

    def validate_model(self, model: str) -> bool:
        """Check if this provider supports the given model.

        Args:
            model: Model name to validate.

        Returns:
            True if model is supported.
        """
        try:
            normalized = self._normalize_model_name(model)
        except ModelProviderError:
            return False

        return normalized.startswith("models/gemini")

    def _normalize_model_name(self, model: str) -> str:
        """Normalize user-supplied model identifiers to Google SDK expected ids."""

        trimmed = (model or "").strip()
        if not trimmed:
            raise ModelProviderError("Model identifier cannot be empty.")

        candidate = trimmed if trimmed.startswith("models/") else f"models/{trimmed}"
        canonical = _MODEL_ALIASES.get(candidate, candidate)

        if not canonical.startswith("models/gemini"):
            raise ModelProviderError(f"Model '{model}' is not a supported Google Gemini model.")

        return canonical

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model.

        Args:
            model: Model name.

        Returns:
            Dictionary with model information.
        """
        info = super().get_model_info(model)

        try:
            canonical = self._normalize_model_name(model)
        except ModelProviderError:
            canonical = model if model.startswith("models/") else f"models/{model}"

        context_window = _MODEL_CONTEXT_WINDOWS.get(canonical)
        if context_window is None:
            if "-flash" in canonical:
                context_window = 1_000_000
            elif "-pro" in canonical or "-ultra" in canonical:
                context_window = 2_000_000
            else:
                context_window = 32768

        supports_vision = any(token in canonical for token in ("-flash", "-pro", "-vision"))

        # Add Google-specific information
        info.update(
            {
                "context_window": context_window,
                "supports_vision": supports_vision,
                "supports_functions": True,  # Gemini supports function calling
                "canonical_model": canonical,
            }
        )

        return info


class GoogleDiscoveryAdapter(DiscoveryProvider):
    """Discovery adapter for Google Gemini models."""

    name = "google"

    def __init__(
        self,
        *,
        api_key_resolver=_resolve_google_api_key,
        module_loader=_load_genai_module,
    ) -> None:
        self._resolve_api_key = api_key_resolver
        self._load_module = module_loader

    def list_models(
        self, *, region: Optional[str] = None, project_hint: Optional[str] = None
    ) -> Iterable[DiscoveredModel]:
        api_key = self._resolve_api_key()
        if not api_key:
            logger.debug("Skipping Google discovery: no API key configured")
            return []

        try:
            module = self._load_module()
        except ModelProviderError as exc:  # pragma: no cover
            logger.debug("Google discovery unavailable: %s", exc)
            return []

        if region:
            logger.debug("Google discovery region hint '%s' not currently applied", region)

        try:
            models = [
                self._to_discovered(model)
                for model in self._iter_models(module, api_key, region, project_hint)
            ]
        except Exception as exc:
            logger.warning("Google discovery failed: %s", exc)
            return []

        return models

    def _iter_models(
        self,
        module: Any,
        api_key: str,
        region: Optional[str],
        project_hint: Optional[str],
    ) -> Iterable[Any]:
        if hasattr(module, "Client"):
            client = module.Client(api_key=api_key)  # type: ignore[no-untyped-call]
            iterator = client.models.list(parent=project_hint)  # type: ignore[arg-type,no-untyped-call]
        else:
            if hasattr(module, "configure"):
                module.configure(api_key=api_key)
            list_kwargs: Dict[str, Any] = {}
            if project_hint:
                list_kwargs["project"] = project_hint
            iterator = module.list_models(**list_kwargs)  # type: ignore[call-arg]

        for model in iterator:
            # Capability filtering (generate/ embed) ensures runnable models.
            capabilities = _extract_sequence(
                model,
                "supported_generation_methods",
                "supportedGenerationMethods",
                "supported_actions",
                "supportedActions",
            )
            if capabilities and not {
                "generateContent",
                "embedContent",
            }.intersection(set(capabilities)):
                continue
            yield model

    def _to_discovered(self, model: Any) -> DiscoveredModel:
        raw_name = (
            getattr(model, "name", None)
            or getattr(model, "model", None)
            or getattr(model, "id", None)
            or ""
        )
        vendor_id = raw_name.split("/")[-1] if raw_name else getattr(model, "id", "")

        display_name = getattr(model, "display_name", None) or getattr(model, "displayName", None)
        description = getattr(model, "description", None)
        context_in = _extract_int(model, "input_token_limit", "inputTokenLimit")
        context_out = _extract_int(model, "output_token_limit", "outputTokenLimit")
        capabilities = tuple(
            _extract_sequence(
                model,
                "supported_generation_methods",
                "supportedGenerationMethods",
                "supported_actions",
                "supportedActions",
            )
        )
        region_scope = tuple(_extract_sequence(model, "supported_regions", "supportedRegions"))
        raw_payload = _extract_payload(model)

        return DiscoveredModel(
            provider="google",
            id=vendor_id,
            display_name=display_name,
            description=description,
            context_window_in=context_in,
            context_window_out=context_out,
            capabilities=capabilities,
            region_scope=region_scope,
            raw_payload=raw_payload,
        )


def _extract_sequence(model: Any, *names: str) -> list[str]:
    for name in names:
        value = getattr(model, name, None)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if item]
        return [str(value)]
    return []


def _extract_int(model: Any, *names: str) -> Optional[int]:
    for name in names:
        value = getattr(model, name, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_payload(model: Any) -> Optional[Dict[str, Any]]:
    to_dict = getattr(model, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
    if isinstance(model, dict):
        return dict(model)
    return None


register_discovery_provider(GoogleDiscoveryAdapter())
