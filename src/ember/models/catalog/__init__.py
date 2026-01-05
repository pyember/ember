"""Curated catalog of language models and provider metadata.

The catalog is the canonical reference that powers runtime discovery, CLI
completion, and documentation snippets across Ember. Each entry captures the
provider, canonical identifier, and context window limits that downstream
components rely on.

Examples:
    >>> from ember.models.catalog import list_available_models
    >>> sorted(list_available_models())[:3]
    ['claude-2.1', 'claude-3-haiku', 'claude-3-haiku-20240307']

"""

from __future__ import annotations

import importlib
import logging
import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from ember._internal.exceptions import ConfigValueError
from ember.models.discovery.dynamic_catalog import DynamicCatalog
from ember.models.discovery.types import ModelKey, OverrideSpec

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Structured metadata describing a catalogued model.

    Attributes:
        id: Canonical identifier used across Ember.
        provider: Provider slug such as 'openai' or 'anthropic'.
        description: Human-readable summary surfaced in UIs and logs.
        context_window: Maximum supported context window in tokens.
        status: Release status flag ('stable', 'preview', 'experimental', or 'deprecated').

    Examples:
        >>> from ember.models.catalog import MODEL_CATALOG
        >>> info = MODEL_CATALOG['gpt-4']
        >>> info.provider
        'openai'
    """

    id: str
    provider: str
    description: str
    context_window: int
    status: str = "stable"
    context_window_out: int | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    region_scope: tuple[str, ...] = field(default_factory=tuple)
    hidden: bool = False
    pricing_override: Mapping[str, Any] | None = None
    discovery_payload: Mapping[str, Any] | None = None


# Complete catalog of available models
MODEL_CATALOG: dict[str, ModelInfo] = {
    # OpenAI Models
    "gpt-4": ModelInfo(
        id="gpt-4",
        provider="openai",
        description="Most capable GPT-4 model",
        context_window=8192,
    ),
    "gpt-4-turbo": ModelInfo(
        id="gpt-4-turbo",
        provider="openai",
        description="GPT-4 Turbo with 128K context",
        context_window=128000,
    ),
    "gpt-4o": ModelInfo(
        id="gpt-4o",
        provider="openai",
        description="Optimized GPT-4 model",
        context_window=128000,
    ),
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        provider="openai",
        description="Small, fast GPT-4 variant",
        context_window=128000,
    ),
    "gpt-3.5-turbo": ModelInfo(
        id="gpt-3.5-turbo",
        provider="openai",
        description="Fast, efficient model",
        context_window=16385,
    ),
    "gpt-3.5-turbo-16k": ModelInfo(
        id="gpt-3.5-turbo-16k",
        provider="openai",
        description="GPT-3.5 with 16K context",
        context_window=16385,
    ),
    # New GPT-4 variants
    "gpt-4.1": ModelInfo(
        id="gpt-4.1",
        provider="openai",
        description="Enhanced GPT-4.1 model",
        context_window=128000,
        status="preview",
    ),
    "gpt-4.1-mini": ModelInfo(
        id="gpt-4.1-mini",
        provider="openai",
        description="Efficient GPT-4.1 mini variant",
        context_window=128000,
        status="preview",
    ),
    "gpt-4.1-nano": ModelInfo(
        id="gpt-4.1-nano",
        provider="openai",
        description="Smallest GPT-4.1 variant",
        context_window=128000,
        status="preview",
    ),
    "gpt-4.5-preview": ModelInfo(
        id="gpt-4.5-preview",
        provider="openai",
        description="Preview of advanced GPT-4.5",
        context_window=128000,
        status="preview",
    ),
    # GPT-5 family
    "gpt-5": ModelInfo(
        id="gpt-5",
        provider="openai",
        description="Flagship GPT-5 model",
        context_window=400000,
        status="preview",
        aliases=("gpt-5-minimal", "gpt-5-low", "gpt-5-medium", "gpt-5-high"),
        capabilities=("tools", "json", "reasoning"),
    ),
    "gpt-5-mini": ModelInfo(
        id="gpt-5-mini",
        provider="openai",
        description="Smaller GPT-5 variant (cost-efficient)",
        context_window=400000,
        status="preview",
        capabilities=("tools", "json", "reasoning"),
    ),
    "gpt-5-nano": ModelInfo(
        id="gpt-5-nano",
        provider="openai",
        description="Smallest GPT-5 variant",
        context_window=400000,
        status="preview",
        capabilities=("tools", "json", "reasoning"),
    ),
    "gpt-5-chat-latest": ModelInfo(
        id="gpt-5-chat-latest",
        provider="openai",
        description="Chat-optimized GPT-5 alias",
        context_window=400000,
        status="preview",
        capabilities=("tools", "json", "reasoning"),
    ),
    "gpt-5-codex": ModelInfo(
        id="gpt-5-codex",
        provider="openai",
        description="GPT-5 model tuned for Codex tooling workflows",
        context_window=400000,
        status="preview",
        capabilities=("tools", "json", "reasoning"),
    ),
    # OpenAI o-series (reasoning models)
    "o1": ModelInfo(
        id="o1",
        provider="openai",
        description="Advanced reasoning model",
        context_window=128000,
        status="preview",
    ),
    "o1-pro": ModelInfo(
        id="o1-pro",
        provider="openai",
        description="Professional reasoning model",
        context_window=128000,
        status="preview",
    ),
    "o1-mini": ModelInfo(
        id="o1-mini",
        provider="openai",
        description="Efficient reasoning model",
        context_window=128000,
        status="preview",
    ),
    "o3": ModelInfo(
        id="o3",
        provider="openai",
        description="Next-generation reasoning model",
        context_window=128000,
        status="preview",
    ),
    "o3-pro": ModelInfo(
        id="o3-pro",
        provider="openai",
        description="Professional o3 reasoning model",
        context_window=128000,
        status="preview",
    ),
    "o3-mini": ModelInfo(
        id="o3-mini",
        provider="openai",
        description="Efficient o3 reasoning model",
        context_window=128000,
        status="preview",
    ),
    "o4-mini": ModelInfo(
        id="o4-mini",
        provider="openai",
        description="Next-generation o4 mini model",
        context_window=128000,
        status="preview",
    ),
    # Anthropic Models
    "claude-opus-4-1-20250805": ModelInfo(
        id="claude-opus-4-1-20250805",
        provider="anthropic",
        description="Claude Opus 4.1 (August 2025 release)",
        context_window=200000,
        status="preview",
    ),
    "claude-opus-4-1": ModelInfo(
        id="claude-opus-4-1",
        provider="anthropic",
        description="Claude Opus 4.1 alias",
        context_window=200000,
        status="preview",
    ),
    "claude-opus-4-20250514": ModelInfo(
        id="claude-opus-4-20250514",
        provider="anthropic",
        description="Claude Opus 4 (May 2025 release)",
        context_window=200000,
        status="preview",
    ),
    "claude-opus-4": ModelInfo(
        id="claude-opus-4",
        provider="anthropic",
        description="Claude Opus 4 alias",
        context_window=200000,
        status="preview",
    ),
    "claude-4-sonnet-20250514": ModelInfo(
        id="claude-4-sonnet-20250514",
        provider="anthropic",
        description="Claude Sonnet 4 (May 2025 release)",
        context_window=200000,
        status="preview",
    ),
    "claude-4-sonnet": ModelInfo(
        id="claude-4-sonnet",
        provider="anthropic",
        description="Claude Sonnet 4 alias",
        context_window=200000,
        status="preview",
        aliases=("claude-sonnet-4",),
    ),
    "claude-sonnet-4-5-20250929": ModelInfo(
        id="claude-sonnet-4-5-20250929",
        provider="anthropic",
        description="Claude Sonnet 4.5 (September 2025 release)",
        context_window=200000,
        status="preview",
        aliases=("claude-4-5-sonnet", "claude-4.5-sonnet", "claude-sonnet-4.5"),
    ),
    "claude-3-7-sonnet-20250219": ModelInfo(
        id="claude-3-7-sonnet-20250219",
        provider="anthropic",
        description="Claude Sonnet 3.7 (February 2025 release)",
        context_window=200000,
        status="preview",
    ),
    "claude-3-7-sonnet-latest": ModelInfo(
        id="claude-3-7-sonnet-latest",
        provider="anthropic",
        description="Claude Sonnet 3.7 latest",
        context_window=200000,
        status="preview",
    ),
    "claude-3-5-sonnet-20241022": ModelInfo(
        id="claude-3-5-sonnet-20241022",
        provider="anthropic",
        description="Claude Sonnet 3.5 (October 2024 release)",
        context_window=200000,
    ),
    "claude-3-5-sonnet-latest": ModelInfo(
        id="claude-3-5-sonnet-latest",
        provider="anthropic",
        description="Claude Sonnet 3.5 latest",
        context_window=200000,
    ),
    "claude-3-5-haiku-20241022": ModelInfo(
        id="claude-3-5-haiku-20241022",
        provider="anthropic",
        description="Claude Haiku 3.5 (October 2024 release)",
        context_window=200000,
    ),
    "claude-3-5-haiku-latest": ModelInfo(
        id="claude-3-5-haiku-latest",
        provider="anthropic",
        description="Claude Haiku 3.5 latest",
        context_window=200000,
    ),
    "claude-3-opus": ModelInfo(
        id="claude-3-opus",
        provider="anthropic",
        description="Claude Opus 3 alias",
        context_window=200000,
    ),
    "claude-3-opus-latest": ModelInfo(
        id="claude-3-opus-latest",
        provider="anthropic",
        description="Claude Opus 3 latest",
        context_window=200000,
    ),
    "claude-3-opus-20240229": ModelInfo(
        id="claude-3-opus-20240229",
        provider="anthropic",
        description="Claude Opus 3 (February 2024 release)",
        context_window=200000,
    ),
    "claude-3-sonnet": ModelInfo(
        id="claude-3-sonnet",
        provider="anthropic",
        description="Claude Sonnet 3 alias",
        context_window=200000,
    ),
    "claude-3-sonnet-20240229": ModelInfo(
        id="claude-3-sonnet-20240229",
        provider="anthropic",
        description="Claude Sonnet 3 (February 2024 release)",
        context_window=200000,
    ),
    "claude-3-haiku": ModelInfo(
        id="claude-3-haiku",
        provider="anthropic",
        description="Claude Haiku 3 alias",
        context_window=200000,
    ),
    "claude-3-haiku-20240307": ModelInfo(
        id="claude-3-haiku-20240307",
        provider="anthropic",
        description="Claude Haiku 3 (March 2024 release)",
        context_window=200000,
    ),
    "claude-2.1": ModelInfo(
        id="claude-2.1",
        provider="anthropic",
        description="Claude 2.1",
        context_window=200000,
        status="deprecated",
    ),
    "claude-instant-1.2": ModelInfo(
        id="claude-instant-1.2",
        provider="anthropic",
        description="Claude Instant 1.2",
        context_window=100000,
        status="deprecated",
    ),
    # Google Models
    "gemini-1.5-pro-latest": ModelInfo(
        id="gemini-1.5-pro-latest",
        provider="google",
        description="Google's Gemini 1.5 Pro (latest) multimodal model",
        context_window=2_000_000,
        capabilities=("vision",),
        aliases=(
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-1.5-pro",
            "models/gemini-pro",
            "models/gemini-pro-vision",
            "models/gemini-1.5-pro",
        ),
    ),
    "gemini-1.5-flash-latest": ModelInfo(
        id="gemini-1.5-flash-latest",
        provider="google",
        description="Google's Gemini 1.5 Flash (latest) fast multimodal model",
        context_window=1_000_000,
        capabilities=("vision",),
        aliases=(
            "gemini-1.5-flash",
            "models/gemini-1.5-flash",
        ),
    ),
    # Gemini 2.5 family (canonical IDs)
    "gemini-2.5-pro": ModelInfo(
        id="gemini-2.5-pro",
        provider="google",
        description="Gemini 2.5 Pro with 1M context",
        context_window=1_000_000,
    ),
    "gemini-2.5-flash": ModelInfo(
        id="gemini-2.5-flash",
        provider="google",
        description="Gemini 2.5 Flash (1M context)",
        context_window=1_000_000,
    ),
    "gemini-2.5-flash-lite": ModelInfo(
        id="gemini-2.5-flash-lite",
        provider="google",
        description="Gemini 2.5 Flash-Lite (1,048,576 tokens)",
        context_window=1_048_576,
    ),
    # Ollama (local) – minimal discovery entries; explicit provider required
    "llama3.2:1b": ModelInfo(
        id="llama3.2:1b",
        provider="ollama",
        description="Local Llama 3.2 1B via Ollama (small, fast to download)",
        context_window=128_000,
        status="preview",
    ),
    "llama3.1:8b": ModelInfo(
        id="llama3.1:8b",
        provider="ollama",
        description="Local Llama 3.1 8B via Ollama",
        context_window=128_000,
        status="preview",
    ),
    "llama3.1:70b": ModelInfo(
        id="llama3.1:70b",
        provider="ollama",
        description="Local Llama 3.1 70B via Ollama",
        context_window=128_000,
        status="preview",
    ),
}


_DYNAMIC_CATALOG = DynamicCatalog(MODEL_CATALOG)
_BOOTSTRAP_FINGERPRINT = id(MODEL_CATALOG)
_CORE_PROVIDER_MODULES: Mapping[str, str] = {
    "anthropic": "ember.models.providers.anthropic",
    "google": "ember.models.providers.google",
    "openai": "ember.models.providers.openai",
}


def _refresh_dynamic_catalog_if_needed() -> None:
    global _DYNAMIC_CATALOG, _BOOTSTRAP_FINGERPRINT
    current = id(MODEL_CATALOG)
    if current != _BOOTSTRAP_FINGERPRINT:
        if isinstance(_DYNAMIC_CATALOG, DynamicCatalog):
            _DYNAMIC_CATALOG = DynamicCatalog(MODEL_CATALOG)
        _BOOTSTRAP_FINGERPRINT = current


_MODEL_INFO_FIELD_NAMES = {field.name for field in fields(ModelInfo)}
_OVERRIDE_FIELDS = {
    "description",
    "pricing",
    "hidden",
    "aliases",
    "capabilities",
    "context_window",
    "context_window_out",
}


def _context_config_value(key: str, default: Any = None) -> Any:
    from ember._internal.context.runtime import EmberContext

    ctx = EmberContext.current()
    return ctx.get_config(key, default)


def _normalize_mode_candidate(value: Any | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in {"live", "dynamic", "auto", "on", "1", "true"}:
        return "live"
    if normalized in {"bootstrap", "static", "cache"}:
        return "bootstrap"
    if normalized in {"off", "disabled", "disable", "0", "false"}:
        return "bootstrap"
    return None


def _resolve_discovery_mode(include_dynamic: bool, explicit: str | None) -> str:
    if not include_dynamic:
        return "bootstrap"

    candidates: Sequence[Any | None] = (
        explicit,
        os.getenv("EMBER_MODEL_DISCOVERY_MODE"),
        _context_config_value("models.discovery_mode"),
        os.getenv("EMBER_MODEL_DISCOVERY"),
    )
    for candidate in candidates:
        normalized = _normalize_mode_candidate(candidate)
        if normalized:
            if normalized != "live":
                logger.debug("Model discovery mode resolved to '%s'", normalized)
            return normalized
    return "live"


def _resolve_overrides() -> Dict[str, OverrideSpec]:
    raw_overrides = _context_config_value("models.overrides", {})
    if raw_overrides is None:
        return {}
    if not isinstance(raw_overrides, Mapping):
        raise ConfigValueError(
            "models.overrides must be a mapping",
            context={"value_type": type(raw_overrides).__name__},
        )
    return _normalize_override_mapping(raw_overrides)


def _normalize_override_mapping(
    overrides: Mapping[str, Any],
    *,
    prefix: str | None = None,
) -> Dict[str, OverrideSpec]:
    normalized: Dict[str, OverrideSpec] = {}
    for key, value in overrides.items():
        if not isinstance(key, str):
            raise ConfigValueError(
                "models.overrides keys must be strings",
                context={"key": key, "value_type": type(key).__name__},
            )
        key_str = key.strip()
        if not key_str:
            raise ConfigValueError("models.overrides keys must be non-empty strings")
        if not isinstance(value, Mapping):
            if prefix is None:
                entry_path = f"models.overrides.{key_str}"
            else:
                entry_path = f"models.overrides.{prefix}.{key_str}"
            raise ConfigValueError(
                "Override entries must be mappings",
                context={"key": entry_path, "value_type": type(value).__name__},
            )
        if prefix is not None:
            provider_key = ModelKey.to_key(prefix, key_str)
            resolved_key, spec = _normalize_override_entry(
                provider_key, value, source_path=f"models.overrides.{prefix}.{key_str}"
            )
            if resolved_key and spec:
                normalized[resolved_key] = spec
            continue
        if ":" in key_str or _contains_override_payload(value):
            resolved_key, spec = _normalize_override_entry(
                key_str, value, source_path=f"models.overrides.{key_str}"
            )
            if resolved_key and spec:
                normalized[resolved_key] = spec
            continue
        nested = _normalize_override_mapping(value, prefix=key_str)
        normalized.update(nested)
    return normalized


def _contains_override_payload(payload: Mapping[str, Any]) -> bool:
    return bool(set(payload) & _OVERRIDE_FIELDS) or "provider" in payload or "id" in payload


def _normalize_override_entry(
    key: str,
    payload: Mapping[str, Any],
    *,
    source_path: str,
) -> tuple[str | None, OverrideSpec | None]:
    provider_key = key
    if ":" not in provider_key:
        provider = payload.get("provider")
        model_id = payload.get("id") or payload.get("model")
        if not isinstance(provider, str) or not provider.strip():
            raise ConfigValueError(
                "Override entry must include provider when key is unqualified",
                context={"key": source_path, "field": "provider", "value": provider},
            )
        if not isinstance(model_id, str) or not model_id.strip():
            raise ConfigValueError(
                "Override entry must include id/model when key is unqualified",
                context={"key": source_path, "field": "id", "value": model_id},
            )
        provider_key = ModelKey.to_key(provider.strip(), model_id.strip())
    else:
        try:
            ModelKey.split(provider_key)
        except ValueError as exc:
            raise ConfigValueError(
                "Override keys must be '<provider>:<id>'",
                context={"key": source_path, "value": provider_key},
            ) from exc

    spec: OverrideSpec = {}
    if "description" in payload:
        description = payload.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ConfigValueError(
                "Override description must be a non-empty string",
                context={"key": source_path, "field": "description", "value": description},
            )
        spec["description"] = description.strip()
    if "pricing" in payload:
        pricing = payload.get("pricing")
        if not isinstance(pricing, Mapping):
            raise ConfigValueError(
                "Override pricing must be a mapping",
                context={"key": source_path, "field": "pricing", "value": pricing},
            )
        pricing_dict: dict[str, Any] = {}
        for pricing_key, pricing_value in pricing.items():
            if not isinstance(pricing_key, str) or not pricing_key.strip():
                raise ConfigValueError(
                    "Override pricing keys must be non-empty strings",
                    context={
                        "key": source_path,
                        "field": "pricing",
                        "pricing_key": pricing_key,
                        "value_type": type(pricing_key).__name__,
                    },
                )
            pricing_dict[pricing_key] = pricing_value
        spec["pricing"] = pricing_dict
    if "hidden" in payload:
        hidden = payload.get("hidden")
        if not isinstance(hidden, bool):
            raise ConfigValueError(
                "Override hidden must be a boolean",
                context={"key": source_path, "field": "hidden", "value": hidden},
            )
        spec["hidden"] = hidden
    if "aliases" in payload:
        aliases = _coerce_string_tuple(
            payload.get("aliases"), source_path=source_path, field="aliases"
        )
        if aliases:
            spec["aliases"] = aliases
    if "capabilities" in payload:
        capabilities = _coerce_string_tuple(
            payload.get("capabilities"), source_path=source_path, field="capabilities"
        )
        if capabilities:
            spec["capabilities"] = capabilities
    if "context_window" in payload:
        spec["context_window"] = _coerce_positive_int(
            payload.get("context_window"),
            source_path=source_path,
            field="context_window",
        )
    if "context_window_out" in payload:
        spec["context_window_out"] = _coerce_positive_int(
            payload.get("context_window_out"),
            source_path=source_path,
            field="context_window_out",
        )

    if not spec:
        return None, None

    return provider_key, spec


def _is_unresolved_placeholder(value: str) -> bool:
    trimmed = value.strip()
    return trimmed.startswith("${") and trimmed.endswith("}")


def _coerce_string_tuple(value: Any, *, source_path: str, field: str) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or _is_unresolved_placeholder(cleaned):
            raise ConfigValueError(
                f"Override {field} must be a non-empty string",
                context={"key": source_path, "field": field, "value": value},
            )
        return (cleaned,)

    if not isinstance(value, (list, tuple, set)):
        raise ConfigValueError(
            f"Override {field} must be a string or sequence of strings",
            context={"key": source_path, "field": field, "value": value},
        )

    cleaned_items: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ConfigValueError(
                f"Override {field} entries must be strings",
                context={"key": source_path, "field": field, "index": idx, "value": item},
            )
        cleaned = item.strip()
        if not cleaned or _is_unresolved_placeholder(cleaned):
            raise ConfigValueError(
                f"Override {field} entries must be non-empty strings",
                context={"key": source_path, "field": field, "index": idx, "value": item},
            )
        cleaned_items.append(cleaned)

    return tuple(cleaned_items)


def _coerce_positive_int(value: Any, *, source_path: str, field: str) -> int:
    if value is None:
        raise ConfigValueError(
            f"Override {field} must be a positive integer",
            context={"key": source_path, "field": field, "value": value},
        )
    if isinstance(value, bool):
        raise ConfigValueError(
            f"Override {field} must be a positive integer (not boolean)",
            context={"key": source_path, "field": field, "value": value},
        )

    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ConfigValueError(
                f"Override {field} must be an integer (no fractional tokens)",
                context={"key": source_path, "field": field, "value": value},
            )
        parsed = int(value)
    elif isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or _is_unresolved_placeholder(cleaned):
            raise ConfigValueError(
                f"Override {field} must be a positive integer",
                context={"key": source_path, "field": field, "value": value},
            )
        try:
            parsed = int(cleaned)
        except ValueError as exc:
            raise ConfigValueError(
                f"Override {field} must be a positive integer",
                context={"key": source_path, "field": field, "value": value},
            ) from exc
    else:
        raise ConfigValueError(
            f"Override {field} must be a positive integer",
            context={"key": source_path, "field": field, "value": value},
        )

    if parsed <= 0:
        raise ConfigValueError(
            f"Override {field} must be positive",
            context={"key": source_path, "field": field, "value": value},
        )
    return parsed


def _merge_aliases(existing: Any, incoming: Iterable[str] | None, fallback: str) -> tuple[str, ...]:
    values: set[str] = set()
    if isinstance(existing, str):
        values.add(existing)
    elif isinstance(existing, (list, tuple, set)):
        values.update(str(item) for item in existing if item)
    if incoming:
        values.update(str(item) for item in incoming if item)
    values.add(fallback)
    return tuple(sorted(values))


def _merge_tuple(existing: Any, incoming: Iterable[str] | None) -> tuple[str, ...]:
    values: set[str] = set()
    if isinstance(existing, str):
        values.add(existing)
    elif isinstance(existing, (list, tuple, set)):
        values.update(str(item) for item in existing if item)
    if incoming:
        values.update(str(item) for item in incoming if item)
    return tuple(sorted(values))


def _build_model_info(data: Mapping[str, Any]) -> ModelInfo:
    payload = dict(data)
    for key in ("aliases", "capabilities", "region_scope"):
        value = payload.get(key)
        if isinstance(value, list):
            payload[key] = tuple(value)
        elif isinstance(value, set):
            payload[key] = tuple(sorted(value))
        elif value is None:
            payload[key] = tuple()
    if not payload.get("id"):
        payload["id"] = str(data.get("id") or "")
    if not payload["id"]:
        raise ValueError("Incomplete model payload, missing {'id'}")

    provider = payload.get("provider")
    if not provider:
        raise ValueError("Incomplete model payload, missing {'provider'}")

    payload.setdefault("description", str(payload["id"]))
    context_in = payload.get("context_window")
    if context_in is None:
        context_in = payload.get("context_window_in")
    payload["context_window"] = int(context_in or 0)
    payload.setdefault("status", "stable")
    filtered = {name: payload.get(name) for name in _MODEL_INFO_FIELD_NAMES if name in payload}
    return ModelInfo(**filtered)  # type: ignore[arg-type]


def _finalize_catalog(merged: Mapping[str, Mapping[str, Any]]) -> Dict[str, ModelInfo]:
    catalog: Dict[str, ModelInfo] = {}
    for record in merged.values():
        canonical_id = record.get("id")
        if not canonical_id:
            continue
        canonical_id = str(canonical_id)
        base = MODEL_CATALOG.get(canonical_id)
        if base:
            if is_dataclass(base):
                data = asdict(base)
            elif isinstance(base, Mapping):
                data = dict(base)
            else:
                raise TypeError(
                    "MODEL_CATALOG entries must be dataclasses or mapping-compatible payloads"
                )
        else:
            provider = record.get("provider") or "unknown"
            description = record.get("description") or canonical_id
            context_in = record.get("context_window") or record.get("context_window_in") or 0
            data = {
                "id": canonical_id,
                "provider": provider,
                "description": description,
                "context_window": int(context_in) if context_in else 0,
                "status": record.get("status", "unknown"),
            }

        data.setdefault("id", canonical_id)

        provider = record.get("provider")
        if provider:
            data["provider"] = provider
        if record.get("description"):
            data["description"] = record["description"]
        if record.get("context_window") is not None:
            data["context_window"] = record["context_window"]
        elif record.get("context_window_in") is not None and not data.get("context_window"):
            data["context_window"] = record["context_window_in"]
        if record.get("context_window_out") is not None:
            data["context_window_out"] = record["context_window_out"]

        data["aliases"] = _merge_aliases(data.get("aliases"), record.get("aliases"), canonical_id)
        data["capabilities"] = _merge_tuple(data.get("capabilities"), record.get("capabilities"))
        data["region_scope"] = _merge_tuple(data.get("region_scope"), record.get("region_scope"))

        if "hidden" in record:
            data["hidden"] = bool(record.get("hidden"))
        if "pricing_override" in record:
            data["pricing_override"] = record.get("pricing_override")
        if "discovery_payload" in record:
            data["discovery_payload"] = record.get("discovery_payload")

        catalog[canonical_id] = _build_model_info(data)

    return catalog


def _build_alias_index(catalog: Mapping[str, ModelInfo]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for canonical, info in catalog.items():
        alias_values = info.aliases or (canonical,)
        for alias in alias_values:
            aliases[str(alias)] = canonical
        aliases[canonical] = canonical
    return aliases


# Static alias support ------------------------------------------------------


_GPT5_REASONING_ALIAS_RE = re.compile(
    r"^(?:(?P<provider>openai)/)?(?P<base>gpt-5)(?:-(?P<effort>minimal|low|medium|high|xhigh))$"
)


def _build_static_alias_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for canonical, info in MODEL_CATALOG.items():
        provider = info.provider or ""
        canonical_lower = canonical.lower()
        if provider:
            provider_lower = provider.lower()
            mapping[f"{provider_lower}/{canonical_lower}"] = canonical
        mapping[canonical_lower] = canonical
        for alias in info.aliases or ():
            alias_text = str(alias)
            alias_lower = alias_text.lower()
            mapping[alias_lower] = canonical
            if provider:
                mapping[f"{provider_lower}/{alias_lower}"] = canonical
    return mapping


_STATIC_ALIAS_MAP = _build_static_alias_map()


def _normalize_display_name(display_name: str) -> str:
    """Normalize display names to canonical model ID format.

    Converts user-friendly display names to Ember's lowercase-hyphenated format.
    Version dots followed by a hyphen become hyphens (Claude-style), while
    trailing version dots are preserved (OpenAI-style).

    Args:
        display_name: Human-readable model name (may contain spaces, mixed case).

    Returns:
        Normalized identifier in lowercase-hyphenated format.
    """
    normalized = display_name.lower().replace(" ", "-").replace("_", "-")
    normalized = re.sub(r"(\d)\.(\d)(?=-)", r"\1-\2", normalized)
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized.strip("-")


def canonicalize_model_identifier(model_id: str) -> tuple[str, dict[str, Any]]:
    """Normalize model identifiers and emit default parameters for aliases.

    Accepts multiple formats and normalizes to Ember's canonical form:
    - Canonical IDs: "gpt-5.1", "claude-opus-4-5-20251101"
    - Display names: "GPT-5.1", "Claude Opus 4.5"
    - Provider-qualified: "openai/gpt-5.1"

    Logs when normalization occurs for debuggability.

    Args:
        model_id: Identifier supplied by callers (may include provider prefix).

    Returns:
        Tuple containing the canonical model identifier and alias default params.
    """

    if not model_id:
        return model_id, {}

    trimmed = model_id.strip()
    if not trimmed:
        return model_id, {}

    key = trimmed.lower()
    match = _GPT5_REASONING_ALIAS_RE.match(key)
    if match:
        canonical = _STATIC_ALIAS_MAP.get(key)
        if canonical is None:
            base_key = match.group("base")
            provider = match.group("provider")
            if provider:
                base_key = f"{provider}/{base_key}"
            canonical = _STATIC_ALIAS_MAP.get(base_key.lower(), trimmed)
        effort = match.group("effort")
        defaults: dict[str, Any] = {}
        if effort:
            defaults["reasoning"] = {"effort": effort}
        return canonical, defaults

    canonical = _STATIC_ALIAS_MAP.get(key)
    if canonical is not None:
        return canonical, {}

    alias_index = _alias_index_for_canonicalize()
    dynamic_canonical = alias_index.get(key)
    if dynamic_canonical is not None:
        return dynamic_canonical, {}

    if " " in trimmed or trimmed != key:
        normalized = _normalize_display_name(trimmed)
        if normalized != key:
            normalized_canonical = _STATIC_ALIAS_MAP.get(normalized)
            if normalized_canonical is not None:
                logger.info("Resolved model '%s' -> '%s'", model_id, normalized_canonical)
                return normalized_canonical, {}
            dynamic_normalized = alias_index.get(normalized)
            if dynamic_normalized is not None:
                logger.info("Resolved model '%s' -> '%s'", model_id, dynamic_normalized)
                return dynamic_normalized, {}
            if normalized != trimmed.lower():
                logger.debug(
                    "Normalized model identifier '%s' -> '%s' (not found in catalog)",
                    model_id,
                    normalized,
                )

    return trimmed, {}


def _alias_index_for_canonicalize() -> Dict[str, str]:
    """Return a case-insensitive alias index including config overrides."""

    _, alias_index = _load_catalog(
        None,
        include_dynamic=False,
        refresh=False,
        discovery_mode="bootstrap",
    )
    lowered = {str(alias).lower(): canonical for alias, canonical in alias_index.items()}

    overrides = _resolve_overrides()
    for key, spec in overrides.items():
        aliases = spec.get("aliases")
        if not aliases:
            continue
        try:
            _, model_id = ModelKey.split(key)
        except ValueError:
            model_id = key
        for alias in aliases:
            lowered[str(alias).lower()] = model_id

    return lowered


def _ensure_core_providers(provider: str | None) -> None:
    """Ensure core discovery providers are registered.

    Raises:
        RuntimeError: If required provider modules cannot be imported.
    """

    target_provider = provider.lower() if provider else None
    if target_provider is None:
        modules: Iterable[str] = _CORE_PROVIDER_MODULES.values()
    else:
        module = _CORE_PROVIDER_MODULES.get(target_provider)
        if module is None:
            return
        modules = (module,)

    import_errors: list[str] = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as exc:
            import_errors.append(f"{module} ({exc})")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to import provider module '{module}' for dynamic discovery"
            ) from exc

    if import_errors:
        errors = ", ".join(import_errors)
        raise RuntimeError(
            "Dynamic model discovery is unavailable because required provider modules "
            f"could not be imported: {errors}. Install the missing provider extras or "
            "call with include_dynamic=False to use the static catalog."
        )


def _load_catalog(
    provider: str | None,
    *,
    include_dynamic: bool,
    refresh: bool,
    discovery_mode: str | None,
) -> Tuple[Dict[str, ModelInfo], Dict[str, str]]:
    _refresh_dynamic_catalog_if_needed()

    overrides = _resolve_overrides()
    mode = _resolve_discovery_mode(include_dynamic, discovery_mode)
    if include_dynamic and mode == "live":
        _ensure_core_providers(provider)
    merged = _DYNAMIC_CATALOG.load(
        provider=provider,
        mode=mode,
        refresh=refresh,
        overrides=overrides,
    )

    catalog = _finalize_catalog(merged)
    if provider:
        catalog = {mid: info for mid, info in catalog.items() if info.provider == provider}

    alias_index = _build_alias_index(catalog)
    return catalog, alias_index


def get_model_overrides() -> Dict[str, OverrideSpec]:
    """Return the normalized model overrides sourced from configuration."""

    return dict(_resolve_overrides())


def list_available_models(
    provider: str | None = None,
    *,
    include_dynamic: bool = True,
    refresh: bool = False,
    discovery_mode: str | None = None,
) -> list[str]:
    """Return the sorted list of known model identifiers."""

    catalog, _ = _load_catalog(
        provider,
        include_dynamic=include_dynamic,
        refresh=refresh,
        discovery_mode=discovery_mode,
    )
    models = [mid for mid, info in catalog.items() if not info.hidden]
    return sorted(models)


def get_providers(
    *, include_dynamic: bool = True, refresh: bool = False, discovery_mode: str | None = None
) -> set[str]:
    """Return the set of provider identifiers present in the catalog."""

    catalog, _ = _load_catalog(
        None,
        include_dynamic=include_dynamic,
        refresh=refresh,
        discovery_mode=discovery_mode,
    )
    return {info.provider for info in catalog.values() if info.provider}


def get_model_info(
    model_id: str,
    *,
    include_dynamic: bool = True,
    refresh: bool = False,
    discovery_mode: str | None = None,
) -> ModelInfo:
    """Look up metadata for a specific model identifier."""

    catalog, alias_index = _load_catalog(
        None,
        include_dynamic=include_dynamic,
        refresh=refresh,
        discovery_mode=discovery_mode,
    )

    canonical = model_id
    if canonical not in catalog:
        canonical = alias_index.get(model_id, model_id)

    if canonical not in catalog:
        available = list(catalog.keys())
        raise KeyError(
            f"Unknown model '{model_id}'. Available models: {', '.join(sorted(available))}"
        )

    return catalog[canonical]


# Model constants for IDE autocomplete
def _sanitize_constant_name(model_id: str) -> str:
    """Transform a model identifier into an exported constant name.

    Args:
        model_id: Model identifier to normalize.

    Returns:
        str: Uppercase constant-style representation.

    Examples:
        >>> _sanitize_constant_name('gpt-4.1-mini')
        'GPT_4_1_MINI'
    """
    out = []
    for ch in model_id:
        if ch.isalnum():
            out.append(ch.upper())
        else:
            out.append("_")
    name = "".join(out)
    # Collapse consecutive underscores
    while "__" in name:
        name = name.replace("__", "_")
    # Trim underscores
    return name.strip("_")


def _build_stable_constants() -> dict[str, str]:
    """Create the mapping of exported model constants.

    Returns:
        dict[str, str]: Mapping from constant name to canonical model id.

    Examples:
        >>> constants = _build_stable_constants()
        >>> constants['GPT_4']
        'gpt-4'
    """
    mapping: dict[str, str] = {}
    for mid, info in MODEL_CATALOG.items():
        if getattr(info, "status", "stable") != "stable":
            continue
        const = _sanitize_constant_name(mid)
        mapping[const] = mid
    return mapping


class _ModelsMeta(type):
    """Resolve attribute lookups for generated model constants.

    Raises:
        AttributeError: Raised when a requested constant is not defined.
    """

    def __getattr__(cls, name: str) -> str:  # type: ignore[override]
        from typing import cast

        mapping = cast(dict[str, str], getattr(cls, "_MAP", {}))
        if name in mapping:
            return mapping[name]
        raise AttributeError(f"Unknown model constant: {name}")

    def __setattr__(cls, name: str, value: object) -> None:  # type: ignore[override]
        # Prevent mutation of exposed constants after creation
        if name in ("_MAP",):
            return super().__setattr__(name, value)
        raise AttributeError("Models is read-only")


class Models(metaclass=_ModelsMeta):
    """Expose stable model identifiers as read-only constants.

    The constants are derived from ``MODEL_CATALOG`` and include only entries
    tagged as ``"stable"``. Preview and experimental releases remain
    discoverable through helper functions rather than constants.

    Examples:
        >>> from ember.models.catalog import Models
        >>> Models.GPT_4
        'gpt-4'
    """


# Attach the generated mapping to the class
Models._MAP = _build_stable_constants()
