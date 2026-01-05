"""Transformation helpers (vmap, pmap, scan, grad) for XCS."""

from __future__ import annotations

import warnings
from collections.abc import Sized
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import tree_util as jax_tree

from ember.xcs.api.jit import jit
from ember.xcs.compiler.analysis import (
    EffectRisk,
    OpKind,
    Traceability,
    analyze_operations,
    analyze_operations_v2,
    has_jax_arrays,
    is_jax_array,
)
from ember.xcs.config import Config
from ember.xcs.errors import XCSError
from ember.xcs.utils.executors import get_shared_executor
from ember.xcs.utils.pytree import StaticWrapper, register_ember_pytrees

CallableT = Callable[..., Any]

_AXIS_CACHE: Dict[
    Tuple[Any, int, Tuple[str, ...]],
    Tuple[Tuple[Optional[int], ...], Tuple[Tuple[str, Optional[int]], ...]],
] = {}

_NUMERIC_SCALARS = (int, float, bool, complex)

register_ember_pytrees()


def vmap(
    func: Optional[CallableT] = None,
    *,
    in_axes: Any = 0,
    out_axes: Any = 0,
    axis_name: Optional[str] = None,
    axis_size: Optional[int] = None,
    config: Optional[Config] = None,
) -> CallableT:
    """Vectorize `func` across the leading axis with orchestration awareness."""

    if func is None:
        return lambda f: vmap(
            f,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            config=config,
        )

    ops = analyze_operations(func)
    effective_config = config or Config()

    if ops.only_tensor_ops:
        jax_vmapped = jax.vmap(
            func,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
        )

        @wraps(func)
        def tensor_wrapper(*args: Any, **kwargs: Any) -> Any:
            if _inputs_can_use_native_jax(args, kwargs):
                return jax_vmapped(*args, **kwargs)
            return _execute_orchestration_batch(
                func,
                args,
                kwargs,
                in_axes,
                out_axes,
                axis_size=axis_size,
                config=effective_config,
            )

        return tensor_wrapper

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return _execute_orchestration_batch(
            func,
            args,
            kwargs,
            in_axes,
            out_axes,
            axis_size=axis_size,
            config=effective_config,
        )

    return wrapper


def pmap(
    func: Optional[CallableT] = None,
    *,
    axis_name: Optional[str] = None,
    in_axes: Any = 0,
    out_axes: Any = 0,
    devices: Optional[Sequence[Any]] = None,
    backend: Optional[str] = None,
    config: Optional[Config] = None,
) -> CallableT:
    """Distributed map that understands orchestration workloads."""

    if func is None:
        return lambda f: pmap(
            f,
            axis_name=axis_name,
            in_axes=in_axes,
            out_axes=out_axes,
            devices=devices,
            backend=backend,
            config=config,
        )

    ops = analyze_operations(func)
    effective_config = config or Config()

    if ops.only_tensor_ops:
        jax_pmapped = jax.pmap(
            func,
            axis_name=axis_name,
            in_axes=in_axes,
            out_axes=out_axes,
            devices=devices,
            backend=backend,
        )

        @wraps(func)
        def tensor_wrapper(*args: Any, **kwargs: Any) -> Any:
            if _inputs_can_use_native_jax(args, kwargs):
                return jax_pmapped(*args, **kwargs)
            return _execute_orchestration_batch(
                func,
                args,
                kwargs,
                in_axes,
                out_axes,
                axis_size=None,
                config=effective_config,
            )

        return tensor_wrapper

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return _execute_orchestration_batch(
            func,
            args,
            kwargs,
            in_axes,
            out_axes,
            axis_size=None,
            config=effective_config,
        )

    return wrapper


def scan(
    func: Optional[CallableT] = None,
    *,
    length: Optional[int] = None,
    reverse: bool = False,
    unroll: int = 1,
) -> CallableT:
    """Sequential scan with orchestration-aware fallback."""

    if func is None:
        return lambda f: scan(f, length=length, reverse=reverse, unroll=unroll)

    ops = analyze_operations(func)

    if ops.only_tensor_ops:

        @wraps(func)
        def tensor_wrapper(
            init: Any, xs: Iterable[Any], *args: Any, **kwargs: Any
        ) -> Tuple[Any, Any]:
            return jax.lax.scan(func, init, xs, length=length, reverse=reverse, unroll=unroll)

        return tensor_wrapper

    @wraps(func)
    def wrapper(init: Any, xs: Iterable[Any], *args: Any, **kwargs: Any) -> Tuple[Any, Any]:
        carry = init
        outputs: List[Any] = []
        sequence = reversed(list(xs)) if reverse else xs
        for item in sequence:
            carry, out = func(carry, item, *args, **kwargs)
            outputs.append(out)
        if reverse:
            outputs.reverse()
        if outputs and hasattr(outputs[0], "shape"):
            stacked = jnp.stack(outputs)
        else:
            stacked = tuple(outputs)
        return carry, stacked

    return wrapper


def grad(
    func: Optional[CallableT] = None,
    *,
    argnums: Union[int, Tuple[int, ...]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    allow_hybrid: bool = False,
    strict: bool = True,
) -> CallableT:
    """Gradient wrapper that rejects pure orchestration workloads.

    Args:
        func: Function to differentiate.
        argnums: Which argument(s) to differentiate with respect to.
        has_aux: Whether function returns (primary, auxiliary) tuple.
        holomorphic: Whether to use holomorphic differentiation.
        allow_int: Whether to allow integer inputs.
        allow_hybrid: If True, allow hybrid functions (tensor + orchestration).
            WARNING: This can execute network calls during JAX tracing.
        strict: If True (default), require explicit markers or pure tensor ops
            for JAX transforms. If False, attempt JAX transforms speculatively.

    Returns:
        Gradient function.

    Raises:
        XCSError: If function is orchestration-only or hybrid (without allow_hybrid).
    """

    if func is None:
        return lambda f: grad(
            f,
            argnums=argnums,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            allow_hybrid=allow_hybrid,
            strict=strict,
        )

    # Use the new rich decision model
    decision = analyze_operations_v2(func)

    # Orchestration-only: always error
    if decision.kind == OpKind.ORCHESTRATION:
        message = "Cannot compute gradients through orchestration-only functions."

        @wraps(func)
        def orchestration_only_wrapper(*args: Any, **kwargs: Any) -> Any:
            raise XCSError(message)

        return orchestration_only_wrapper

    # Unknown: error in strict mode, warn in lenient mode
    if decision.kind == OpKind.UNKNOWN:
        if strict:
            message = (
                "Cannot compute gradients through unknown function. "
                "Add @mark_tensor to enable JAX transforms."
            )

            @wraps(func)
            def unknown_wrapper(*args: Any, **kwargs: Any) -> Any:
                raise XCSError(message)

            return unknown_wrapper
        else:
            warnings.warn(
                f"grad() on unknown function {getattr(func, '__name__', repr(func))}; "
                "behavior may be unpredictable",
                UserWarning,
                stacklevel=2,
            )

    # Hybrid: require explicit opt-in
    if decision.kind == OpKind.HYBRID:
        if not allow_hybrid:
            message = (
                "Cannot compute gradients through hybrid (tensor + orchestration) functions. "
                "This would execute orchestration calls during JAX tracing. "
                "Either split the function or pass allow_hybrid=True if you understand the risks."
            )

            @wraps(func)
            def hybrid_wrapper(*args: Any, **kwargs: Any) -> Any:
                raise XCSError(message)

            return hybrid_wrapper
        else:
            warnings.warn(
                f"grad() on hybrid function {getattr(func, '__name__', repr(func))}; "
                "orchestration calls may execute during tracing",
                UserWarning,
                stacklevel=2,
            )

    # Tensor-only with non-JAX traceability: error
    if (
        decision.kind == OpKind.TENSOR
        and decision.jax_traceable == Traceability.NON_JAX
    ):
        message = (
            "Cannot compute gradients through non-JAX tensor functions (torch/numpy/tf). "
            "Use jax.numpy instead of numpy for JAX compatibility."
        )

        @wraps(func)
        def non_jax_wrapper(*args: Any, **kwargs: Any) -> Any:
            raise XCSError(message)

        return non_jax_wrapper

    # High effect risk: error
    if decision.effect_risk == EffectRisk.HIGH:
        message = (
            "Cannot compute gradients through effectful functions (I/O, network, subprocess). "
            "JAX tracing would execute side effects multiple times."
        )

        @wraps(func)
        def effectful_wrapper(*args: Any, **kwargs: Any) -> Any:
            raise XCSError(message)

        return effectful_wrapper

    # Safe path: tensor-only with JAX traceability
    if decision.kind == OpKind.TENSOR and decision.jax_traceable == Traceability.JAX:
        return jax.grad(
            func,
            argnums=argnums,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
        )

    # Neutral or tensor with unknown traceability: attempt JAX
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        def safe_func(*inner_args: Any, **inner_kwargs: Any) -> Any:
            result = func(*inner_args, **inner_kwargs)
            if has_aux:
                primary, aux = result
                return _sanitize_non_array_leaves(primary), _sanitize_non_array_leaves(aux)
            return _sanitize_non_array_leaves(result)

        hybrid_grad = jax.grad(
            safe_func,
            argnums=argnums,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
        )
        result = hybrid_grad(*args, **kwargs)
        if has_aux:
            grads, aux = result
            return grads, _unwrap_static_wrappers(aux)
        return result

    return wrapper


# Helpers -----------------------------------------------------------------


def _execute_orchestration_batch(
    func: CallableT,
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    in_axes: Any,
    out_axes: Any,
    *,
    axis_size: Optional[int],
    config: Config,
) -> Any:
    pos_axes, kw_axes = _normalize_axes(in_axes, len(args), tuple(kwargs.keys()))
    batch_size = _infer_batch_size(args, kwargs, pos_axes, kw_axes, axis_size)
    if batch_size is None or batch_size <= 0:
        return func(*args, **kwargs)

    def _evaluate(index: int) -> Any:
        item_args, item_kwargs = _slice_arguments(
            args,
            kwargs,
            pos_axes,
            kw_axes,
            index,
        )
        return func(*item_args, **item_kwargs)

    if not config.parallel:
        results = [_evaluate(index) for index in range(batch_size)]
        return _stack_results(results, out_axes)

    worker_limit = config.max_workers if config.max_workers is not None else min(32, batch_size)
    max_workers = min(worker_limit, batch_size)

    executor = get_shared_executor(max_workers)
    futures = [executor.submit(_evaluate, index) for index in range(batch_size)]
    results = [future.result() for future in futures]

    return _stack_results(results, out_axes)


def _normalize_axes(
    in_axes: Any,
    num_positional: int,
    kw_names: Tuple[str, ...],
) -> Tuple[Tuple[Optional[int], ...], Dict[str, Optional[int]]]:
    key = (_freeze_in_axes(in_axes), num_positional, kw_names)
    cached = _AXIS_CACHE.get(key)
    if cached is not None:
        pos_axes_cached, kw_axes_cached = cached
        return pos_axes_cached, dict(kw_axes_cached)

    if isinstance(in_axes, (list, tuple)):
        pos_axes = list(in_axes) + [None] * max(0, num_positional - len(in_axes))
        pos_axes = pos_axes[:num_positional]
        kw_axes = {name: None for name in kw_names}
    elif isinstance(in_axes, dict):
        pos_axes = [None] * num_positional
        kw_axes = {name: in_axes.get(name) for name in kw_names}
    elif isinstance(in_axes, int):
        pos_axes = [in_axes] * num_positional
        kw_axes = {name: in_axes for name in kw_names}
    elif in_axes is None:
        pos_axes = [None] * num_positional
        kw_axes = {name: None for name in kw_names}
    else:
        raise XCSError(f"Unsupported in_axes specification: {in_axes!r}")

    pos_axes_tuple = tuple(pos_axes)
    kw_axes_tuple = tuple((name, axis) for name, axis in kw_axes.items())
    _AXIS_CACHE[key] = (pos_axes_tuple, kw_axes_tuple)
    return pos_axes_tuple, kw_axes


def _infer_batch_size(
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    pos_axes: Tuple[Optional[int], ...],
    kw_axes: Mapping[str, Optional[int]],
    axis_size: Optional[int],
) -> Optional[int]:
    candidate_sizes: List[int] = []

    for axis, value in zip(pos_axes, args, strict=False):
        length = _axis_length(value, axis)
        if length is not None:
            candidate_sizes.append(length)

    for name, axis in kw_axes.items():
        if name not in kwargs:
            continue
        length = _axis_length(kwargs[name], axis)
        if length is not None:
            candidate_sizes.append(length)

    if axis_size is not None:
        if candidate_sizes:
            reference = candidate_sizes[0]
            if any(size != reference for size in candidate_sizes):
                raise XCSError("Mismatched batch dimensions across arguments.")
            if reference > axis_size:
                raise XCSError(
                    "Input batch size exceeds declared axis_size for orchestration fallback."
                )
            return min(reference, axis_size)
        return axis_size

    if not candidate_sizes:
        return None

    reference = candidate_sizes[0]
    if any(size != reference for size in candidate_sizes):
        raise XCSError("Mismatched batch dimensions across arguments.")
    return reference


def _slice_arguments(
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    pos_axes: Tuple[Optional[int], ...],
    kw_axes: Mapping[str, Optional[int]],
    index: int,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    sliced_args = tuple(
        _slice_value(value, axis, index) for axis, value in zip(pos_axes, args, strict=False)
    )

    sliced_kwargs: Dict[str, Any] = {}
    for name, axis in kw_axes.items():
        if name in kwargs:
            sliced_kwargs[name] = _slice_value(kwargs[name], axis, index)
    for name, value in kwargs.items():
        sliced_kwargs.setdefault(name, value)

    return sliced_args, sliced_kwargs


def _axis_length(value: Any, axis: Optional[int]) -> Optional[int]:
    if axis is None or value is None:
        return None
    if hasattr(value, "shape"):
        dims = value.shape
        rank = len(dims)
        normalized = axis if axis >= 0 else rank + axis
        if normalized < 0 or normalized >= rank:
            raise XCSError("in_axes axis is out of bounds for argument shape.")
        return int(dims[normalized])
    if axis != 0:
        raise XCSError(
            "Only axis=0 is supported for non-array arguments in orchestration fallback."
        )
    if isinstance(value, Sized):
        return len(value)
    return None


def _slice_value(value: Any, axis: Optional[int], index: int) -> Any:
    if axis is None or value is None:
        return value
    if not hasattr(value, "__getitem__"):
        raise XCSError("Cannot slice value for orchestration batching.")
    if hasattr(value, "shape"):
        dims = value.shape
        rank = len(dims)
        normalized = axis if axis >= 0 else rank + axis
        if normalized < 0 or normalized >= rank:
            raise XCSError("in_axes axis is out of bounds for argument shape.")
        slicer = [slice(None)] * rank
        slicer[normalized] = index
        return value[tuple(slicer)]
    if axis != 0:
        raise XCSError(
            "Only axis=0 is supported for non-array arguments in orchestration fallback."
        )
    return value[index]


def _freeze_in_axes(spec: Any) -> Any:
    if isinstance(spec, dict):
        return ("dict", tuple(sorted((key, _freeze_in_axes(value)) for key, value in spec.items())))
    if isinstance(spec, (list, tuple)):
        return (type(spec).__name__, tuple(_freeze_in_axes(item) for item in spec))
    return spec


def _stack_results(results: List[Any], out_axes: Any) -> Any:
    if not results:
        return results

    sample = results[0]
    if isinstance(sample, Mapping):
        sample_leaves, sample_treedef = jax_tree.tree_flatten(sample)

        has_non_numeric = any(
            not _is_array_like(leaf) and not isinstance(leaf, _NUMERIC_SCALARS)
            for leaf in sample_leaves
        )
        if has_non_numeric:
            return [dict(item) if isinstance(item, Mapping) else item for item in results]
    else:
        sample_leaves, sample_treedef = jax_tree.tree_flatten(sample)

    if not sample_leaves:
        return results

    batched_leaves: List[List[Any]] = []
    for item in results:
        leaves, item_treedef = jax_tree.tree_flatten(item)
        if item_treedef != sample_treedef:
            raise XCSError("Inconsistent output structure across orchestration batch results.")
        batched_leaves.append(leaves)

    axis_leaves = _broadcast_axes_leaves(out_axes, sample_treedef)
    if len(axis_leaves) != len(sample_leaves):
        raise XCSError("out_axes specification does not match output structure.")

    stacked_leaves = []
    for leaf_values, axis_spec in zip(
        zip(*batched_leaves, strict=False), axis_leaves, strict=False
    ):
        stacked_leaves.append(_stack_leaf(list(leaf_values), axis_spec))

    stacked = jax_tree.tree_unflatten(sample_treedef, stacked_leaves)
    if isinstance(sample, Mapping):
        return jax_tree.tree_map(_convert_array_like_to_list, stacked)
    return stacked


def _broadcast_axes_leaves(out_axes: Any, treedef: jax_tree.PyTreeDef) -> List[Any]:
    leaves: List[Any] = []

    def _fill(axis_spec: Any, current_def: jax_tree.PyTreeDef) -> None:
        if jax_tree.tree_structure(axis_spec) == current_def:
            axis_leaves = current_def.flatten_up_to(axis_spec)
            for leaf in axis_leaves:
                if not isinstance(leaf, (int, type(None))):
                    raise XCSError("out_axes values must be integers or None.")
                leaves.append(leaf)
            return

        if jax_tree.treedef_is_leaf(current_def):
            if isinstance(axis_spec, (int, type(None))):
                leaves.append(axis_spec)
                return
            raise XCSError("out_axes values must be integers or None.")

        if isinstance(axis_spec, (int, type(None))):
            for child_def in current_def.children():
                _fill(axis_spec, child_def)
            return

        child_defs = current_def.children()
        _, metadata = current_def.node_data()

        if isinstance(axis_spec, Mapping):
            if metadata is None:
                raise XCSError("out_axes dict can only target mapping-like outputs.")
            missing = set(metadata) - set(axis_spec.keys())
            extra = set(axis_spec.keys()) - set(metadata)
            if missing or extra:
                raise XCSError("out_axes dict must match output dict keys.")
            for key, child_def in zip(metadata, child_defs, strict=False):
                _fill(axis_spec[key], child_def)
            return

        if isinstance(axis_spec, (list, tuple)):
            if len(axis_spec) != len(child_defs):
                raise XCSError("out_axes sequence length must match output structure.")
            for item_spec, child_def in zip(axis_spec, child_defs, strict=False):
                _fill(item_spec, child_def)
            return

        raise XCSError("Unsupported out_axes specification for output structure.")

    _fill(out_axes, treedef)
    return leaves


def _stack_leaf(values: List[Any], axis_spec: Any) -> Any:
    if axis_spec is None:
        return _collapse_leaf(values)
    if not isinstance(axis_spec, int):
        raise XCSError("out_axes values must be integers or None.")

    if not any(_is_array_like(value) for value in values):
        if axis_spec in (0, -1):
            return list(values)

    try:
        stacked = jnp.stack(values, axis=0)
    except Exception as exc:
        if axis_spec not in (0, -1):
            raise XCSError(
                "Cannot reorder outputs for non-array leaves with out_axes != 0."
            ) from exc
        return values

    target_axis = axis_spec if axis_spec >= 0 else stacked.ndim + axis_spec
    if target_axis < 0 or target_axis >= stacked.ndim:
        raise XCSError("out_axes axis is out of bounds for output leaf.")
    if target_axis == 0:
        return stacked
    return jnp.moveaxis(stacked, 0, target_axis)


def _collapse_leaf(values: List[Any]) -> Any:
    reference = values[0]
    for value in values[1:]:
        if not _leaf_equal(reference, value):
            raise XCSError(
                "out_axes=None requires identical outputs across the orchestration batch."
            )
    return reference


def _leaf_equal(lhs: Any, rhs: Any) -> bool:
    if hasattr(lhs, "shape") and hasattr(rhs, "shape"):
        try:
            return bool(jnp.array_equal(lhs, rhs))
        except Exception:
            return False
    return lhs == rhs


def _sanitize_non_array_leaves(value: Any) -> Any:
    if isinstance(value, StaticWrapper):
        return value
    if is_jax_array(value):
        return value

    try:
        leaves, treedef = jax_tree.tree_flatten(value)
    except TypeError:
        return _wrap_static_leaf(value)

    sanitized = [_wrap_static_leaf(leaf) for leaf in leaves]
    return jax_tree.tree_unflatten(treedef, sanitized)


def _wrap_static_leaf(value: Any) -> Any:
    if isinstance(value, StaticWrapper):
        return value
    if is_jax_array(value):
        return value
    return StaticWrapper(value)


def _unwrap_static_wrappers(value: Any) -> Any:
    if isinstance(value, StaticWrapper):
        return value.value
    try:
        return jax_tree.tree_map(
            lambda leaf: leaf.value if isinstance(leaf, StaticWrapper) else leaf,
            value,
            is_leaf=lambda leaf: isinstance(leaf, StaticWrapper),
        )
    except Exception:
        return value


def _inputs_can_use_native_jax(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> bool:
    if not has_jax_arrays(args, kwargs):
        return False

    try:
        leaves = jax_tree.tree_leaves((args, dict(kwargs)))
    except Exception:
        return False

    for leaf in leaves:
        if isinstance(leaf, StaticWrapper):
            return False
        if is_jax_array(leaf):
            continue
        if isinstance(leaf, _NUMERIC_SCALARS) or leaf is None:
            continue
        return False
    return True


def _is_array_like(value: Any) -> bool:
    if is_jax_array(value):
        return True
    return hasattr(value, "__array__")


def _convert_array_like_to_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except TypeError:
            return value
    return value


__all__ = ["jit", "vmap", "pmap", "scan", "grad"]
