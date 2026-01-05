"""Dynamic catalog orchestration for provider-backed discovery.

This module centralises runtime fusion of live discovery data, the curated
bootstrap catalog, and user-supplied overrides. The implementation keeps the
logic isolated from the higher-level catalog helpers so other modules can
request merged metadata without worrying about provider wiring, caching, or
fallback rules.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import is_dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

from ember.models.discovery.merge import MergedCatalog, ModelInfoLike, merge_catalog
from ember.models.discovery.registry import get_provider, list_providers
from ember.models.discovery.types import DiscoveredModel, OverrideSpec

logger = logging.getLogger(__name__)


class DynamicCatalog:
    """Load and cache model metadata discovered from external providers.

    The catalog merges three layers of data:
      1. Bootstrap definitions curated in the repository
      2. Live discovery results returned by provider APIs
      3. User overrides supplied through configuration or environment hooks
    """

    _CACHE_MISS: Dict[str, DiscoveredModel] = {}

    def __init__(
        self,
        bootstrap: Mapping[str, ModelInfoLike],
        *,
        fingerprint_resolver: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._bootstrap = dict(bootstrap)
        self._cache: Dict[Tuple[str, str, str, str], Dict[str, DiscoveredModel]] = {}
        self._lock = threading.RLock()
        self._fingerprint_resolver = fingerprint_resolver or self._default_fingerprint

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(
        self,
        *,
        provider: Optional[str] = None,
        mode: str = "live",
        refresh: bool = False,
        overrides: Optional[Mapping[str, OverrideSpec]] = None,
        region: Optional[str] = None,
        project_hint: Optional[str] = None,
    ) -> MergedCatalog:
        """Return the merged catalog for the requested scope."""

        normalized_mode = self._normalize_mode(mode)
        overrides = overrides or {}

        bootstrap_view = self._filtered_bootstrap(provider)
        if normalized_mode == "bootstrap":
            dynamic: Mapping[str, DiscoveredModel] = {}
        else:
            dynamic = self._discover(
                provider=provider,
                refresh=refresh,
                region=region,
                project_hint=project_hint,
            )

        return merge_catalog(
            dynamic=dynamic,
            bootstrap=bootstrap_view,
            overrides=overrides,
        )

    def clear_cache(self) -> None:
        """Drop cached discovery results."""

        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _discover(
        self,
        *,
        provider: Optional[str],
        refresh: bool,
        region: Optional[str],
        project_hint: Optional[str],
    ) -> Dict[str, DiscoveredModel]:
        names: Iterable[str]
        if provider:
            names = (provider,)
        else:
            names = list_providers()

        discovered: Dict[str, DiscoveredModel] = {}
        for name in names:
            try:
                adapter = get_provider(name)
            except KeyError:
                continue

            cache_key = self._cache_key(name, region, project_hint)
            if not refresh:
                cached = self._cache_lookup(cache_key)
                if cached is not None:
                    discovered.update(cached)
                    continue

            try:
                models = adapter.list_models(region=region, project_hint=project_hint)
            except Exception as exc:  # pragma: no cover
                logger.warning("Model discovery failed for provider %s: %s", name, exc)
                self._store_cache(cache_key, self._CACHE_MISS)
                continue

            provider_models: Dict[str, DiscoveredModel] = {}
            for model in models:
                if not isinstance(model, DiscoveredModel):
                    continue
                provider_models[model.model_key()] = model

            self._store_cache(cache_key, provider_models)
            discovered.update(provider_models)

        return discovered

    def _filtered_bootstrap(self, provider: Optional[str]) -> Dict[str, ModelInfoLike]:
        if provider is None:
            return dict(self._bootstrap)
        filtered: Dict[str, ModelInfoLike] = {}
        for model_id, info in self._bootstrap.items():
            candidate_provider = self._bootstrap_provider(info)
            if candidate_provider == provider:
                filtered[model_id] = info
        return filtered

    def _bootstrap_provider(self, info: ModelInfoLike) -> Optional[str]:
        if is_dataclass(info):
            return getattr(info, "provider", None)
        if isinstance(info, Mapping):
            provider = info.get("provider")
            if isinstance(provider, str):
                return provider
        return None

    def _cache_lookup(self, key: Tuple[str, str, str, str]) -> Optional[Dict[str, DiscoveredModel]]:
        with self._lock:
            cached = self._cache.get(key)
            if cached is self._CACHE_MISS:
                return {}
            if cached is None:
                return None
            return dict(cached)

    def _store_cache(
        self,
        key: Tuple[str, str, str, str],
        value: Dict[str, DiscoveredModel],
    ) -> None:
        with self._lock:
            if not value:
                self._cache[key] = self._CACHE_MISS
            else:
                self._cache[key] = dict(value)

    def _cache_key(
        self,
        provider: str,
        region: Optional[str],
        project_hint: Optional[str],
    ) -> Tuple[str, str, str, str]:
        fingerprint = self._fingerprint_resolver(provider)
        return (
            provider,
            fingerprint,
            region or "",
            project_hint or "",
        )

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = (mode or "live").strip().lower()
        if normalized in {"off", "disabled", "disable"}:
            return "bootstrap"
        if normalized in {"bootstrap", "static", "cache"}:
            return "bootstrap"
        return "live"

    @staticmethod
    def _default_fingerprint(provider: str) -> str:
        """Return a non-secret fingerprint for provider configuration.

        Fingerprints are derived from configuration values (never raw secrets)
        and are used only to invalidate discovery caches when a provider's
        access context changes.
        """

        tokens = _fingerprint_provider_config(provider)
        if not tokens:
            return f"{provider}:anon"
        digest = hashlib.sha256("|".join(tokens).encode("utf-8")).hexdigest()[:12]
        return f"{provider}:{digest}"


__all__ = ["DynamicCatalog"]


def _fingerprint_provider_config(provider: str) -> list[str]:
    from ember._internal.context.runtime import EmberContext

    ctx = EmberContext.current()
    tokens: list[str] = []

    api_key = ctx.get_config(f"providers.{provider}.api_key")
    if isinstance(api_key, str):
        cleaned = api_key.strip()
        if cleaned and not (cleaned.startswith("${") and cleaned.endswith("}")):
            tokens.append(_hash_value(f"providers.{provider}.api_key", cleaned))

    base_url = ctx.get_config(f"providers.{provider}.base_url")
    if isinstance(base_url, str) and base_url.strip():
        tokens.append(_hash_value(f"providers.{provider}.base_url", base_url.strip()))

    return tokens


def _hash_value(label: str, value: str) -> str:
    digest = hashlib.sha256(value.strip().encode("utf-8")).hexdigest()[:12]
    return f"{label}:{digest}"
