"""Execution engine for XCS compiled graphs."""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
import types
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ember.xcs.compiler.graph import GraphParallelismAnalysis, IRGraph, IRNode
from ember.xcs.config import Config
from ember.xcs.errors import XCSError, XCSExecutionError
from ember.xcs.runtime.broker import (
    NODE_METADATA_KEY,
    CapacityBroker,
    Reservation,
    ReservationHint,
    get_capacity_broker,
)

logger = logging.getLogger(__name__)

_BROKER_LOG_SUPPRESSION_WINDOW_SECONDS = 60.0
_MISSING = object()


def _make_output_stub(
    output_vars: Tuple[Tuple[str, ...], ...],
    context: "ExecutionContext",
    *,
    node_id: str,
    name: str,
) -> Callable[..., Any]:
    """Return a stub that reads dependency outputs from the current context."""
    iterator = iter(output_vars)

    def _stub(*_args: Any, **_kwargs: Any) -> Any:
        try:
            outputs = next(iterator)
        except StopIteration as exc:
            raise XCSError(
                f"XCS stub exhausted for '{name}' while executing node {node_id}; "
                "the cached graph no longer matches the current control flow."
            ) from exc
        if len(outputs) == 1:
            return context.get(outputs[0])
        return tuple(context.get(output_name) for output_name in outputs)

    return _stub


@dataclass(slots=True)
class ExecutionContext:
    """Mutable execution state shared across node evaluations."""

    variables: MutableMapping[str, Any] = field(default_factory=dict)

    def set(self, name: str, value: Any) -> None:
        self.variables[name] = value

    def get(self, name: str) -> Any:
        value = self.variables.get(name, _MISSING)
        if value is _MISSING:
            raise XCSError(f"XCS execution missing required variable '{name}'.")
        return value

    def snapshot(self) -> "ExecutionContext":
        return ExecutionContext(variables=dict(self.variables))


class ExecutionEngine:
    """Execute IR graphs sequentially or with coarse-grained parallelism."""

    def __init__(self, max_workers: int = 4, broker: CapacityBroker | None = None) -> None:
        self._default_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._broker = broker
        self._broker_warning_state: Dict[str, float] = {}
        self._broker_warning_lock = threading.Lock()

    def execute(
        self,
        graph: IRGraph,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        *,
        parallelism: Optional[GraphParallelismAnalysis] = None,
        config: Optional[Config] = None,
    ) -> Any:
        context = ExecutionContext()
        for index, arg in enumerate(args):
            context.set(f"arg_{index}", arg)
        for name, value in kwargs.items():
            context.set(name, value)

        selected = config or Config()
        can_parallelize = (
            selected.parallel
            and parallelism is not None
            and parallelism.estimated_speedup > 1.5
            and bool(parallelism.parallel_groups)
        )
        if not can_parallelize:
            return self._execute_sequential(graph, context)
        return self._execute_parallel(graph, context, selected, parallelism)

    def shutdown(self) -> None:
        self._default_executor.shutdown(wait=True)

    def _execute_sequential(self, graph: IRGraph, context: ExecutionContext) -> Any:
        result = None
        for node_id in graph.topological_order():
            result = self._run_node(graph.nodes[node_id], context)
        return result

    def _execute_parallel(
        self,
        graph: IRGraph,
        context: ExecutionContext,
        config: Config,
        parallelism: GraphParallelismAnalysis,
    ) -> Any:
        group_lookup = {
            node_id: group for group in parallelism.parallel_groups for node_id in group
        }
        futures: Dict[str, concurrent.futures.Future[Tuple[Any, Dict[str, Any]]]] = {}

        executor, owns_executor = self._select_executor(config)
        try:
            for node_id in graph.topological_order():
                node = graph.nodes[node_id]
                deps = graph.get_dependencies(node_id)
                for dep in deps:
                    if dep in futures:
                        try:
                            _, outputs = futures.pop(dep).result()
                        except Exception:
                            self._cancel_pending(futures)
                            raise
                        for name, value in outputs.items():
                            context.set(name, value)
                group = group_lookup.get(node_id)
                if group and len(group) > 1 and not node.metadata.get("is_return"):
                    futures[node_id] = executor.submit(
                        self._run_node_isolated, node, context.snapshot()
                    )
                else:
                    self._run_node(node, context)

            while futures:
                node_id, future = futures.popitem()
                try:
                    _, outputs = future.result()
                except Exception:
                    self._cancel_pending(futures)
                    raise
                for name, value in outputs.items():
                    context.set(name, value)
        finally:
            if owns_executor:
                executor.shutdown(wait=True)

        return context.get("return_value")

    def _cancel_pending(
        self, futures: Dict[str, concurrent.futures.Future[Any]]
    ) -> None:
        """Cancel all pending futures. Best-effort, does not raise."""
        for future in futures.values():
            future.cancel()
        futures.clear()

    def _select_executor(
        self, config: Config
    ) -> Tuple[concurrent.futures.ThreadPoolExecutor, bool]:
        if config.max_workers:
            return concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers), True
        return self._default_executor, False

    def _run_node(self, node: IRNode, context: ExecutionContext) -> Any:
        args, kwargs = self._resolve_call(node, context)

        reservation: Reservation | None = None
        inject_kwarg: str | None = None
        broker = self._broker or get_capacity_broker()
        hint, inject_kwarg = self._extract_reservation_metadata(node)
        if hint and broker:
            try:
                reservation = broker.reserve(hint)
            except Exception as exc:  # pragma: no cover
                if self._should_log_broker_failure(node.id):
                    logger.warning(
                        "Capacity broker failed to reserve for node %s",
                        node.id,
                        exc_info=True,
                        extra={
                            "node_id": node.id,
                            "hint": self._serialize_hint(hint),
                            "exc_class": exc.__class__.__name__,
                        },
                    )
                reservation = None

        operator_to_call = node.operator
        stubbed_funcs = (
            node.metadata.get("stubbed_funcs") if isinstance(node.metadata, dict) else None
        )
        if stubbed_funcs:
            operator_to_call = self._prepare_stubbed_operator(node, stubbed_funcs, context)

        if reservation is not None and inject_kwarg:
            kwargs = dict(kwargs)
            kwargs.setdefault(inject_kwarg, reservation)

        try:
            result = operator_to_call(*args, **kwargs)
        except XCSError:
            # Don't double-wrap XCS errors
            if reservation is not None:
                try:
                    broker.release(reservation, success=False)
                except Exception as release_exc:
                    logger.error(
                        "Capacity broker release failure for node %s",
                        node.id,
                        exc_info=release_exc,
                    )
            raise
        except Exception as exc:
            if reservation is not None:
                try:
                    broker.release(reservation, success=False)
                except Exception as release_exc:
                    logger.error(
                        "Capacity broker release failure for node %s",
                        node.id,
                        exc_info=release_exc,
                    )
            raise XCSExecutionError(
                node.id, "operator execution failed", cause=exc
            ) from exc
        else:
            if reservation is not None:
                try:
                    broker.release(reservation, success=True)
                except Exception as release_exc:
                    logger.error(
                        "Capacity broker release failure for node %s",
                        node.id,
                        exc_info=release_exc,
                    )

        if node.outputs:
            values = self._normalize_outputs(node, result)
            for name, value in zip(node.outputs, values, strict=False):
                context.set(name, value)
        if node.metadata.get("is_return"):
            context.set("return_value", result)
        return result

    def _should_log_broker_failure(self, node_id: str) -> bool:
        now = time.monotonic()
        with self._broker_warning_lock:
            last = self._broker_warning_state.get(node_id)
            if last is None or now - last >= _BROKER_LOG_SUPPRESSION_WINDOW_SECONDS:
                self._broker_warning_state[node_id] = now
                return True
            return False

    def _serialize_hint(self, hint: ReservationHint | None) -> Dict[str, Any]:
        if hint is None:
            return {}
        return {
            "logical_model": hint.logical_model,
            "candidate_providers": hint.candidate_providers,
            "concurrency_key": hint.concurrency_key,
            "metadata": dict(hint.metadata),
        }

    def _extract_reservation_metadata(
        self, node: IRNode
    ) -> tuple[ReservationHint | None, str | None]:
        raw = node.metadata.get(NODE_METADATA_KEY)
        if raw is None:
            return None, None

        inject_kwarg: str | None = None
        hint_value: ReservationHint | Mapping[str, Any] | None

        if isinstance(raw, ReservationHint):
            hint_value = raw
        elif isinstance(raw, Mapping):
            candidate_hint = raw.get("hint")
            if isinstance(candidate_hint, ReservationHint):
                hint_value = candidate_hint
            else:
                hint_value = raw
            possible_kwarg = raw.get("inject_kwarg")
            if isinstance(possible_kwarg, str) and possible_kwarg:
                inject_kwarg = possible_kwarg
        else:
            return None, None

        hint = self._coerce_hint(hint_value)
        if hint is None:
            return None, None
        return hint, inject_kwarg

    def _coerce_hint(
        self, value: ReservationHint | Mapping[str, Any] | None
    ) -> ReservationHint | None:
        if value is None:
            return None
        if isinstance(value, ReservationHint):
            return value
        if isinstance(value, Mapping):
            data = dict(value)
            logical = data.get("logical_model")
            if not isinstance(logical, str) or not logical:
                return None
            candidates = data.get("candidate_providers")
            if isinstance(candidates, (list, tuple)):
                candidate_tuple = tuple(str(item) for item in candidates if item)
            else:
                candidate_tuple = tuple()
            concurrency_key = data.get("concurrency_key")
            if concurrency_key is not None and not isinstance(concurrency_key, str):
                concurrency_key = str(concurrency_key)
            metadata_value = data.get("metadata")
            metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
            return ReservationHint(
                logical_model=logical,
                candidate_providers=candidate_tuple,
                concurrency_key=concurrency_key,
                metadata=dict(metadata),
            )
        return None

    def _run_node_isolated(
        self, node: IRNode, context: ExecutionContext
    ) -> Tuple[Any, Dict[str, Any]]:
        result = self._run_node(node, context)
        outputs = {name: context.get(name) for name in node.outputs}
        if node.metadata.get("is_return"):
            outputs["return_value"] = context.get("return_value")
        return result, outputs

    def _prepare_stubbed_operator(
        self,
        node: IRNode,
        stubbed_funcs: Sequence[Mapping[str, Any]],
        context: ExecutionContext,
    ) -> Callable[..., Any]:
        globals_dict = getattr(node.operator, "__globals__", None)
        if not isinstance(globals_dict, dict):
            return node.operator

        operator_func = node.operator
        if not isinstance(operator_func, types.FunctionType):
            return node.operator

        grouped: Dict[str, List[Tuple[str, ...]]] = {}
        module_name = getattr(node.operator, "__module__", None)
        for entry in stubbed_funcs:
            name = entry.get("name")
            if not name or name not in globals_dict:
                continue
            if entry.get("module") != module_name:
                continue
            outputs = entry.get("outputs")
            if not isinstance(outputs, tuple) or not outputs:
                raise XCSError(
                    f"Invalid stub metadata for '{name}' while executing node {node.id}."
                )

            func_obj = entry.get("func")
            current_obj = globals_dict[name]
            if current_obj is not func_obj:
                current_code = getattr(current_obj, "__code__", None)
                target_code = getattr(func_obj, "__code__", None)
                if current_code is None or current_code is not target_code:
                    continue
            grouped.setdefault(name, []).append(outputs)

        if not grouped:
            return node.operator

        isolated_globals = dict(globals_dict)
        for name, sequence in grouped.items():
            isolated_globals[name] = _make_output_stub(
                tuple(sequence),
                context,
                node_id=node.id,
                name=name,
            )

        return types.FunctionType(
            operator_func.__code__,
            isolated_globals,
            operator_func.__name__,
            operator_func.__defaults__,
            operator_func.__closure__,
        )

    def _resolve_call(
        self, node: IRNode, context: ExecutionContext
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        metadata = node.metadata
        if "call_spec_pos" not in metadata:
            raise RuntimeError(
                f"Node {node.id} missing call_spec_pos metadata; graph construction must supply it."
            )
        pos = metadata.get("call_spec_pos", ())
        kwargs_spec = metadata.get("call_spec_kw", {})
        args = tuple(self._resolve_spec(spec, context) for spec in pos)
        kw = {name: self._resolve_spec(spec, context) for name, spec in kwargs_spec.items()}
        return args, kw

    def _resolve_spec(self, spec: Mapping[str, Any], context: ExecutionContext) -> Any:
        kind = spec.get("kind")
        if kind in {"var", "arg", "kwarg"}:
            return context.get(spec.get("name"))
        if kind in {"literal", "value"}:
            return spec.get("value")
        raise ValueError(f"Unsupported call-spec kind '{kind}' encountered during execution.")

    def _normalize_outputs(self, node: IRNode, result: Any) -> Tuple[Any, ...]:
        if not node.outputs:
            return tuple()
        if len(node.outputs) == 1:
            return (result,)
        if isinstance(result, tuple) and len(result) == len(node.outputs):
            return result
        try:
            values = tuple(result)
        except TypeError as exc:  # pragma: no cover
            raise RuntimeError(
                f"Node {node.id} expected {len(node.outputs)} outputs but operator"
                " returned a non-iterable result."
            ) from exc
        if len(values) != len(node.outputs):
            raise RuntimeError(
                f"Node {node.id} produced {len(values)} outputs but graph"
                f" expects {len(node.outputs)}."
            )
        return values


__all__ = ["ExecutionEngine", "ExecutionContext"]
