"""Decorator helpers for wrapping plain callables as Ember operators."""

from __future__ import annotations

import inspect
import threading
from dataclasses import FrozenInstanceError
from functools import update_wrapper
from typing import Any, Callable, Dict, Generic, Optional, ParamSpec, TypeVar, cast

from ember.operators.base import Operator

P = ParamSpec("P")
R = TypeVar("R")

_CONFIGURABLE_FIELDS = {"input_spec", "output_spec"}


class _FunctionOperatorProxy(Generic[P, R]):
    """Lazily materializes a function-based operator with progressive disclosure."""

    _is_operator_proxy = True

    def __init__(
        self,
        fn: Callable[P, R],
        *,
        operator_factory: Callable[[], Operator],
        signature: inspect.Signature,
        input_spec: Optional[type] = None,
        output_spec: Optional[type] = None,
        name: Optional[str] = None,
    ) -> None:
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_factory", operator_factory)
        object.__setattr__(self, "_signature", signature)
        object.__setattr__(
            self, "_pending_config", self._build_initial_config(input_spec, output_spec)
        )
        object.__setattr__(self, "_instance", None)
        object.__setattr__(self, "_lock", threading.Lock())
        update_wrapper(self, fn)  # type: ignore[arg-type]
        object.__setattr__(self, "__signature__", signature)
        if name:
            object.__setattr__(self, "__name__", name)

    @staticmethod
    def _build_initial_config(
        input_spec: Optional[type], output_spec: Optional[type]
    ) -> Dict[str, Optional[type]]:
        config: Dict[str, Optional[type]] = {}
        if input_spec is not None:
            config["input_spec"] = input_spec
        if output_spec is not None:
            config["output_spec"] = output_spec
        return config

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name in _CONFIGURABLE_FIELDS:
            if self._instance is not None:
                raise FrozenInstanceError(
                    "Cannot modify validation specs after operator materialization."
                )
            self._pending_config[name] = value
            return
        instance = self._instance
        if instance is None:
            raise AttributeError(f"Attribute '{name}' is not available before first call.")
        setattr(instance, name, value)

    def __getattr__(self, name: str) -> Any:
        if name in _CONFIGURABLE_FIELDS:
            if self._instance is None:
                return self._pending_config.get(name)
            return getattr(self._instance, name)
        instance = self._instance
        if instance is None:
            raise AttributeError(f"Attribute '{name}' is not available before first call.")
        return getattr(instance, name)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        operator = self._materialize()
        return cast(R, operator(*args, **kwargs))

    def _materialize(self) -> Operator:
        instance = self._instance
        if instance is not None:
            if isinstance(instance, Operator):
                return instance
            raise TypeError(f"Expected Operator, got {type(instance).__name__}")
        with self._lock:
            instance = self._instance
            if instance is None:
                instance = self._factory()
                if not isinstance(instance, Operator):
                    raise TypeError(
                        f"Factory returned {type(instance).__name__}, expected Operator"
                    )
                for field, value in self._pending_config.items():
                    if value is not None:
                        object.__setattr__(instance, field, value)
                signature = object.__getattribute__(self, "_signature")
                object.__setattr__(instance, "__signature__", signature)
                object.__setattr__(instance, "__wrapped__", self._fn)
                object.__setattr__(self, "_instance", instance)
        if not isinstance(instance, Operator):
            raise TypeError(f"Expected Operator, got {type(instance).__name__}")
        return instance

    def __repr__(self) -> str:
        fn = object.__getattribute__(self, "_fn")
        return f"FunctionOperatorProxy({fn.__qualname__})"


def op(
    fn: Optional[Callable[P, R]] = None,
    *,
    input_spec: Optional[type] = None,
    output_spec: Optional[type] = None,
    name: Optional[str] = None,
) -> Callable[..., Any] | Operator:
    """Wrap a Python callable in an :class:`~ember.operators.base.Operator`.

    Args:
        fn: Callable to expose as an operator. Positional and keyword arguments
            are forwarded unchanged.
        input_spec: Optional validation type applied before invocation.
        output_spec: Optional validation type applied to the result.
        name: Optional override for the operator name in logs/graphs.

    Returns:
        A lazily materialized operator facade whose signature mirrors ``fn`` and
        participates in Ember's operator ecosystem.
    """

    if fn is None:
        return lambda wrapped: op(
            wrapped, input_spec=input_spec, output_spec=output_spec, name=name
        )

    fn_signature = inspect.signature(fn)
    # Capture fn in closure to avoid None check issues
    captured_fn = fn

    class FunctionOperator(Operator):  # type: ignore[misc]
        """Operator implementation that forwards calls to ``fn``."""

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return captured_fn(*args, **kwargs)

    FunctionOperator.__name__ = f"FunctionOperator({fn.__name__})"
    FunctionOperator.__doc__ = fn.__doc__
    if hasattr(fn, "__annotations__"):
        FunctionOperator.forward.__annotations__ = dict(fn.__annotations__)

    def operator_factory() -> Operator:
        instance = FunctionOperator()
        if not isinstance(instance, Operator):
            raise TypeError(f"FunctionOperator is not an Operator: {type(instance)}")
        object.__setattr__(instance, "_function_signature", fn_signature)
        object.__setattr__(instance, "__doc__", fn.__doc__)
        object.__setattr__(instance, "__module__", fn.__module__)
        object.__setattr__(instance, "__qualname__", getattr(fn, "__qualname__", fn.__name__))
        return instance

    return _FunctionOperatorProxy(
        fn,
        operator_factory=operator_factory,
        signature=fn_signature,
        input_spec=input_spec,
        output_spec=output_spec,
        name=name,
    )


def mark_orchestration(func: Callable[P, R]) -> Callable[P, R]:
    """Mark a callable as orchestration-oriented for XCS graph heuristics.

    Use this decorator on functions that perform LLM/API calls to ensure
    XCS correctly classifies them and avoids invalid JAX transformations.

    Example:
        @mark_orchestration
        def call_llm(prompt: str) -> str:
            return client.complete(prompt)
    """
    func._xcs_is_orchestration = True  # type: ignore[attr-defined]
    func._xcs_is_tensor = False  # type: ignore[attr-defined]
    return func


def mark_tensor(func: Callable[P, R]) -> Callable[P, R]:
    """Mark a callable as tensor-only for XCS graph heuristics.

    Use this decorator on functions that perform pure numerical/tensor
    operations to enable native JAX transformations (jit, vmap, grad).

    Example:
        @mark_tensor
        def compute_loss(x: jnp.ndarray, y: jnp.ndarray) -> float:
            return jnp.mean((x - y) ** 2)
    """
    func._xcs_is_tensor = True  # type: ignore[attr-defined]
    func._xcs_is_orchestration = False  # type: ignore[attr-defined]
    return func


def mark_hybrid(func: Callable[P, R]) -> Callable[P, R]:
    """Mark a callable as hybrid (tensor + orchestration) for XCS heuristics.

    Use this decorator on functions that mix numerical operations with
    LLM/API calls. XCS will use appropriate strategies for each component.

    Example:
        @mark_hybrid
        def embed_and_query(text: str) -> jnp.ndarray:
            embedding = encode(text)  # tensor op
            result = llm_call(text)    # orchestration
            return process(embedding, result)
    """
    func._xcs_is_tensor = True  # type: ignore[attr-defined]
    func._xcs_is_orchestration = True  # type: ignore[attr-defined]
    return func


def mark_pytree_safe(safe: bool = True) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a callable's pytree safety for vmap/pmap eligibility.

    Use this decorator to explicitly control whether XCS considers a function's
    outputs suitable for JAX pytree operations (vmap batching, etc.).

    Args:
        safe: If True, function outputs are valid JAX pytrees. If False,
            outputs contain non-pytree values (generators, file handles, etc.).

    Example:
        @mark_pytree_safe(False)
        def get_file_handle(path: str) -> IO:
            return open(path)

        @mark_pytree_safe(True)
        def compute_stats(data: jnp.ndarray) -> Dict[str, float]:
            return {"mean": float(data.mean()), "std": float(data.std())}
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        func._xcs_is_pytree_safe = safe  # type: ignore[attr-defined]
        return func
    return decorator


__all__ = ["op", "mark_orchestration", "mark_tensor", "mark_hybrid", "mark_pytree_safe"]
