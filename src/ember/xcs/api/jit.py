"""Public JIT decorator for XCS."""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import os
import threading
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
    overload,
)

from ember.xcs.api.state import ANALYZER, BUILDER, ENGINE, PROFILER
from ember.xcs.compiler.graph import GraphParallelismAnalysis, IRGraph
from ember.xcs.config import Config
from ember.xcs.errors import XCSError

P = ParamSpec("P")
R = TypeVar("R")


logger = logging.getLogger(__name__)


_TENSOR_MODULE_PREFIXES: Tuple[str, ...] = (
    "jax.",
    "jaxlib",
    "numpy",
    "scipy",
    "cupy",
    "tensorflow",
    "torch",
    "pytorch",
    "mxnet",
)


@dataclass(slots=True)
class _CacheEntry:
    graph: IRGraph
    analysis: GraphParallelismAnalysis
    key: str
    config_signature: Tuple[Tuple[str, Any], ...]
    is_pure_tensor: bool


@overload
def jit(__func: Callable[P, R], *, config: Config | None = ...) -> Callable[P, R]: ...


@overload
def jit(*, config: Config | None = ...) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def jit(
    func: Optional[Callable[P, R]] = None,
    *,
    config: Optional[Config] = None,
) -> Callable[..., Any]:
    """Return a function wrapped with the XCS JIT."""

    if func is None:
        return lambda wrapped_func: jit(wrapped_func, config=config)

    if hasattr(func, "_xcs_wrapped"):
        return func

    signature = inspect.signature(func)
    if "_xcs_config" in signature.parameters:
        raise XCSError("Parameter '_xcs_config' is reserved for XCS decorators.")
    accepts_config = "config" in signature.parameters

    cache_lock = threading.Lock()
    cache: Dict[str, _CacheEntry] = {}
    pure_graph_detected = False
    stats_state: Dict[str, Any] = {
        "hits": 0,
        "misses": 0,
        "cache_key_failures": 0,
        "pure_tensor_classify_failures": 0,
        "last_analysis": None,
    }

    @functools.wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:  # type: ignore[misc]
        nonlocal pure_graph_detected
        call_kwargs: Dict[str, Any] = dict(kwargs)
        override_value = _extract_config_override(call_kwargs, accepts_config)
        baseline = config or Config()
        effective_config = _resolve_effective_config(baseline, override_value)

        if pure_graph_detected:
            start = time.perf_counter()
            result = func(*args, **call_kwargs)  # type: ignore[misc]
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if _should_profile(effective_config):
                PROFILER.record(func.__name__, elapsed_ms, None)
            return result

        cache_key: Optional[str] = None
        entry: Optional[_CacheEntry] = None
        if effective_config.cache:
            cache_key = _make_cache_key(func, args, call_kwargs, effective_config)
            if cache_key is None:
                with cache_lock:
                    stats_state["cache_key_failures"] += 1
            else:
                with cache_lock:
                    entry = cache.get(cache_key)
                    if entry is not None:
                        stats_state["hits"] += 1

        if entry is None:
            start = time.perf_counter()
            graph, traced_result = BUILDER.trace_with_result(func, *args, **call_kwargs)
            analysis = ANALYZER.analyze(graph)
            try:
                is_pure_tensor = _graph_is_pure_tensor(graph)
            except Exception:  # pragma: no cover - fail-closed optimization gate
                with cache_lock:
                    stats_state["pure_tensor_classify_failures"] += 1
                logger.debug(
                    "Pure-tensor classification failed for %s; treating as orchestration graph",
                    func.__name__,
                    exc_info=True,
                )
                is_pure_tensor = False
            with cache_lock:
                stats_state["misses"] += 1
                stats_state["last_analysis"] = analysis
                if effective_config.cache and cache_key is not None:
                    cache[cache_key] = _CacheEntry(
                        graph=graph,
                        analysis=analysis,
                        key=cache_key,
                        config_signature=_config_signature(effective_config),
                        is_pure_tensor=is_pure_tensor,
                    )
            if is_pure_tensor:
                pure_graph_detected = True
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if _should_profile(effective_config):
                PROFILER.record(func.__name__, elapsed_ms, None if is_pure_tensor else analysis)
            return traced_result  # type: ignore[no-any-return]

        graph = entry.graph
        analysis = entry.analysis
        is_pure_tensor = entry.is_pure_tensor
        with cache_lock:
            stats_state["last_analysis"] = analysis

        start = time.perf_counter()
        try:
            result = ENGINE.execute(
                graph,
                args,
                call_kwargs,
                parallelism=analysis,
                config=effective_config,
            )
        except TypeError as exc:
            if cache_key is not None:
                with cache_lock:
                    cache.pop(cache_key, None)
            logger.debug(
                "Graph execution raised TypeError for %s; attempting direct call fallback",
                func.__name__,
                exc_info=True,
            )
            try:
                return func(*args, **call_kwargs)
            except Exception as direct_exc:
                raise direct_exc from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if _should_profile(effective_config):
            PROFILER.record(func.__name__, elapsed_ms, analysis)
        return result

    def stats() -> Dict[str, Any]:
        with cache_lock:
            cache_size = len(cache)
            hits = stats_state["hits"]
            misses = stats_state["misses"]
            cache_key_failures = stats_state["cache_key_failures"]
            pure_tensor_classify_failures = stats_state["pure_tensor_classify_failures"]
            analysis = stats_state["last_analysis"]
        if analysis is None:
            return {
                "status": "cold",
                "optimized": False,
                "entries": cache_size,
                "hits": hits,
                "misses": misses,
                "cache_key_failures": cache_key_failures,
                "pure_tensor_classify_failures": pure_tensor_classify_failures,
            }
        optimized = bool(analysis.parallel_groups) and analysis.estimated_speedup > 1.2
        payload: Dict[str, Any] = {
            "status": "compiled" if optimized else "fallback",
            "optimized": optimized,
            "parallel_groups": len(analysis.parallel_groups),
            "estimated_speedup": analysis.estimated_speedup,
            "entries": cache_size,
            "hits": hits,
            "misses": misses,
            "cache_key_failures": cache_key_failures,
            "pure_tensor_classify_failures": pure_tensor_classify_failures,
        }
        if not optimized:
            payload["reason"] = "no_parallelism"
        return payload

    wrapped._xcs_wrapped = True  # type: ignore[attr-defined]
    wrapped._xcs_original = func  # type: ignore[attr-defined]
    wrapped.stats = stats  # type: ignore[attr-defined]
    return wrapped


def _should_profile(config: Config) -> bool:
    if config.profile:
        return True
    return os.environ.get("EMBER_XCS_PROFILE") == "1"


def _extract_config_override(call_kwargs: Dict[str, Any], accepts_config: bool) -> Optional[Any]:
    """Return a per-call config override, if any."""

    internal_value: Optional[Any] = None
    for key in ("_xcs_config", "_config"):
        if key in call_kwargs:
            value = call_kwargs.pop(key)
            if internal_value is None:
                internal_value = value
            else:
                warnings.warn(
                    "Multiple internal config overrides provided; using the first value.",
                    stacklevel=2,
                )

    has_user_config = "config" in call_kwargs
    user_value: Optional[Any] = None
    if not accepts_config and has_user_config:
        user_value = call_kwargs.pop("config")

    if internal_value is not None and has_user_config:
        warnings.warn(
            "Both 'config' and '_xcs_config' were provided; '_xcs_config' takes precedence.",
            stacklevel=2,
        )

    if internal_value is not None:
        return internal_value
    if not accepts_config:
        return user_value
    return None


def _resolve_effective_config(default: Config, override: Optional[Any]) -> Config:
    if override is None:
        return default
    if isinstance(override, Config):
        return override
    if isinstance(override, Mapping):
        return default.apply_overrides(override)
    raise XCSError("Config overrides must be a Config instance or a mapping of configuration keys.")


def _make_cache_key(
    func: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    config: Config,
) -> Optional[str]:
    arg_hashes: list[str] = []
    for arg in args:
        hashed = _hash_value(arg)
        if hashed is None:
            return None
        arg_hashes.append(hashed)

    kw_hashes: list[tuple[str, str]] = []
    for name, value in kwargs.items():
        hashed = _hash_value(value)
        if hashed is None:
            return None
        kw_hashes.append((name, hashed))

    payload = (
        id(func),
        getattr(func, "__module__", ""),
        getattr(func, "__qualname__", getattr(func, "__name__", "unknown")),
        tuple(arg_hashes),
        tuple(sorted(kw_hashes)),
        _config_signature(config),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _hash_value(value: Any) -> Optional[str]:
    encoded = _encode_cache_value(value, visited=set())
    if encoded is None:
        return None
    return hashlib.sha256(encoded).hexdigest()


def _encode_cache_value(value: Any, *, visited: set[int]) -> Optional[bytes]:
    if value is None:
        return _pack_cache_bytes(b"N", b"")
    if isinstance(value, bool):
        return _pack_cache_bytes(b"B", b"\x01" if value else b"\x00")
    if isinstance(value, int):
        return _pack_cache_bytes(b"I", str(value).encode("utf-8"))
    if isinstance(value, float):
        return _pack_cache_bytes(b"F", value.hex().encode("utf-8"))
    if isinstance(value, str):
        return _pack_cache_bytes(b"S", value.encode("utf-8"))
    if isinstance(value, bytes):
        return _pack_cache_bytes(b"Y", value)

    if isinstance(value, tuple):
        return _encode_cache_sequence(value, tag=b"T", visited=visited)
    if isinstance(value, list):
        return _encode_cache_sequence(value, tag=b"L", visited=visited)
    if isinstance(value, dict):
        return _encode_cache_mapping(value, visited=visited)
    if isinstance(value, (set, frozenset)):
        return _encode_cache_set(value, visited=visited)

    return None


def _encode_cache_sequence(
    value: Sequence[Any],
    *,
    tag: bytes,
    visited: set[int],
) -> Optional[bytes]:
    value_id = id(value)
    if value_id in visited:
        return None
    visited.add(value_id)
    try:
        parts: list[bytes] = [len(value).to_bytes(8, "big")]
        for item in value:
            encoded = _encode_cache_value(item, visited=visited)
            if encoded is None:
                return None
            parts.append(encoded)
        return _pack_cache_bytes(tag, b"".join(parts))
    finally:
        visited.remove(value_id)


def _encode_cache_mapping(value: dict[Any, Any], *, visited: set[int]) -> Optional[bytes]:
    value_id = id(value)
    if value_id in visited:
        return None
    visited.add(value_id)
    try:
        if not all(isinstance(key, str) for key in value):
            return None

        parts: list[bytes] = [len(value).to_bytes(8, "big")]
        for key in sorted(value):
            parts.append(_pack_cache_bytes(b"K", key.encode("utf-8")))
            encoded_value = _encode_cache_value(value[key], visited=visited)
            if encoded_value is None:
                return None
            parts.append(encoded_value)
        return _pack_cache_bytes(b"D", b"".join(parts))
    finally:
        visited.remove(value_id)


def _encode_cache_set(value: set[Any] | frozenset[Any], *, visited: set[int]) -> Optional[bytes]:
    value_id = id(value)
    if value_id in visited:
        return None
    visited.add(value_id)
    try:
        elements: list[bytes] = []
        for item in value:
            encoded = _encode_cache_value(item, visited=visited)
            if encoded is None:
                return None
            elements.append(encoded)
        elements.sort()
        payload = len(elements).to_bytes(8, "big") + b"".join(elements)
        tag = b"R" if isinstance(value, frozenset) else b"E"
        return _pack_cache_bytes(tag, payload)
    finally:
        visited.remove(value_id)


def _pack_cache_bytes(tag: bytes, payload: bytes) -> bytes:
    return tag + len(payload).to_bytes(8, "big") + payload


def _config_signature(config: Config) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(config.to_dict().items()))


def get_jit_stats(func: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """Return profiler statistics for `func` or all functions."""
    if func is None:
        return PROFILER.summary()
    return PROFILER.get(getattr(func, "__name__", "unknown"))


def _graph_is_pure_tensor(graph: IRGraph) -> bool:
    """Return True when the graph contains only tensor-oriented operations."""

    has_tensor_ops = False
    for node in graph.nodes.values():
        if node.metadata.get("is_orchestration"):
            return False
        module = getattr(node.operator, "__module__", "") or ""
        if module.startswith(_TENSOR_MODULE_PREFIXES):
            has_tensor_ops = True
    return has_tensor_ops


__all__ = ["jit", "get_jit_stats"]
