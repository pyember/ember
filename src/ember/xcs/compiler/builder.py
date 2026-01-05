"""Convert traced Python execution into IR graphs."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple

from ember.xcs.compiler.graph import IRGraph, IRNode, node_from_callable
from ember.xcs.runtime.broker import NODE_METADATA_KEY
from ember.xcs.tracing.python_tracer import OperationRecord, PythonTracer, TracingError


@dataclass(slots=True)
class _BuilderState:
    nodes: MutableMapping[str, IRNode] = field(default_factory=dict)
    value_to_var: MutableMapping[int, str] = field(default_factory=dict)
    node_counter: int = 0
    var_counter: int = 0
    record_outputs: List[Tuple[str, ...]] = field(default_factory=list)
    record_funcs: List[Callable[..., Any]] = field(default_factory=list)
    record_results: List[Any] = field(default_factory=list)
    record_node_ids: List[str] = field(default_factory=list)

    def next_node_id(self) -> str:
        self.node_counter += 1
        return f"node_{self.node_counter}"

    def next_var(self) -> str:
        self.var_counter += 1
        return f"var_{self.var_counter}"


class IRBuilder:
    """Build immutable IR graphs from runtime traces."""

    def __init__(self, tracer: Optional[PythonTracer] = None) -> None:
        self._tracer = tracer or PythonTracer()
        self.tracer = self._tracer  # Backwards compatibility
        self._state_local = threading.local()

    def trace(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> IRGraph:
        """Trace `func` with `args`/`kwargs` and return an `IRGraph`."""
        try:
            records = self._tracer.trace_function(func, *args, **kwargs)
        except TracingError:
            return self._opaque_graph(func, args, kwargs)

        if not records:
            return self._opaque_graph(func, args, kwargs)

        self._state = _BuilderState()
        arg_bindings = {id(arg): f"arg_{index}" for index, arg in enumerate(args)}
        kw_bindings = {id(value): name for name, value in kwargs.items()}

        try:
            if len(records) == 1:
                self._add_record(0, records[0], arg_bindings, kw_bindings, is_return=True)
                return self._finalize()

            *inner_records, final_record = records
            for index, record in enumerate(inner_records):
                self._add_record(index, record, arg_bindings, kw_bindings, is_return=False)

            self._add_record(
                len(inner_records), final_record, arg_bindings, kw_bindings, is_return=True
            )

            return self._finalize()
        finally:
            del self._state

    def trace_with_result(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Tuple[IRGraph, Any]:
        """Trace `func` with `args`/`kwargs` and return `(IRGraph, result)`."""
        if getattr(func, "_is_operator_proxy", False):
            result = func(*args, **kwargs)
            return self._opaque_graph(func, args, kwargs), result

        try:
            records = self._tracer.trace_function(func, *args, **kwargs)
        except TracingError:
            result = func(*args, **kwargs)
            return self._opaque_graph(func, args, kwargs), result

        if not records:
            result = func(*args, **kwargs)
            return self._opaque_graph(func, args, kwargs), result

        result = records[-1].result
        self._state = _BuilderState()
        arg_bindings = {id(arg): f"arg_{index}" for index, arg in enumerate(args)}
        kw_bindings = {id(value): name for name, value in kwargs.items()}

        try:
            if len(records) == 1:
                self._add_record(0, records[0], arg_bindings, kw_bindings, is_return=True)
                return self._finalize(), result

            *inner_records, final_record = records
            for index, record in enumerate(inner_records):
                self._add_record(index, record, arg_bindings, kw_bindings, is_return=False)

            self._add_record(
                len(inner_records), final_record, arg_bindings, kw_bindings, is_return=True
            )

            return self._finalize(), result
        finally:
            del self._state

    def trace_function(
        self, func: Callable[..., Any], args: Tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> IRGraph:
        """Legacy alias for compatibility with previous API."""
        return self.trace(func, *args, **kwargs)

    # Internal helpers --------------------------------------------------
    def _opaque_graph(
        self, func: Callable[..., Any], args: Tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> IRGraph:
        self._state = _BuilderState()
        try:
            node_id = self._state.next_node_id()
            arg_inputs = tuple(f"arg_{index}" for index, _ in enumerate(args))
            kw_inputs = tuple(kwargs.keys())
            inputs = arg_inputs + kw_inputs
            output = self._state.next_var()
            call_spec_pos = tuple({"kind": "arg", "name": name} for name in arg_inputs)
            metadata = {
                "opaque": True,
                "call_spec_pos": call_spec_pos,
                "call_spec_kw": {name: {"kind": "kwarg", "name": name} for name in kw_inputs},
            }
            node = node_from_callable(node_id, func, inputs, (output,), metadata)
            self._state.nodes[node_id] = node
            return self._finalize()
        finally:
            del self._state

    def _add_record(
        self,
        record_index: int,
        record: OperationRecord,
        arg_bindings: Mapping[int, str],
        kw_bindings: Mapping[int, str],
        *,
        is_return: bool = False,
    ) -> None:
        node_id = self._state.next_node_id()
        inputs: List[str] = []
        call_spec_pos: List[Dict[str, Any]] = []
        call_spec_kw: Dict[str, Dict[str, Any]] = {}

        raw_args = getattr(record, "args", ())
        if raw_args is None:
            raw_args = ()
        elif not isinstance(raw_args, (list, tuple)):
            raw_args = (raw_args,)

        for value in raw_args:
            var_name, spec = self._argument_spec(value, arg_bindings, kw_bindings)
            if var_name:
                inputs.append(var_name)
            call_spec_pos.append(spec)

        raw_kwargs = getattr(record, "kwargs", {})
        if isinstance(raw_kwargs, dict):
            kw_items = raw_kwargs.items()
        elif hasattr(raw_kwargs, "items"):
            try:
                kw_items = list(raw_kwargs.items())
            except TypeError:
                kw_items = []
        else:
            kw_items = []

        for name, value in kw_items:
            var_name, spec = self._argument_spec(
                value,
                arg_bindings,
                kw_bindings,
                keyword=name,
            )
            if var_name:
                inputs.append(var_name)
            call_spec_kw[name] = spec

        outputs = self._allocate_outputs(record.result)

        self._state.record_outputs.append(outputs)
        self._state.record_funcs.append(record.func)
        self._state.record_results.append(record.result)
        self._state.record_node_ids.append(node_id)

        dependencies = tuple(getattr(record, "dependencies", ()))

        for dep_index in dependencies:
            if dep_index < len(self._state.record_outputs):
                for output_name in self._state.record_outputs[dep_index]:
                    if output_name not in inputs:
                        inputs.append(output_name)

        metadata = {
            "dependencies": dependencies,
            "call_spec_pos": tuple(call_spec_pos),
            "call_spec_kw": dict(call_spec_kw),
        }

        dispatch_metadata = self._dispatch_metadata(record)
        if dispatch_metadata is not None:
            metadata[NODE_METADATA_KEY] = dispatch_metadata

        if dependencies:
            stub_entries: List[Dict[str, Any]] = []
            for dep_index in dependencies:
                if dep_index >= len(self._state.record_node_ids):
                    continue
                dep_node_id = self._state.record_node_ids[dep_index]
                dep_node = self._state.nodes.get(dep_node_id)
                if dep_node is None:
                    continue
                func = dep_node.operator
                func_name = getattr(func, "__name__", None)
                module_name = getattr(func, "__module__", "") or ""
                if not func_name or not hasattr(func, "__globals__"):
                    continue
                if not (
                    module_name.startswith("ember.")
                    or module_name.startswith("tests.")
                    or module_name.startswith("test")
                    or module_name.startswith("__main__")
                ):
                    continue
                stub_entries.append(
                    {
                        "name": func_name,
                        "func": func,
                        "module": module_name,
                        "outputs": self._state.record_outputs[dep_index],
                    }
                )
            if stub_entries:
                metadata["stubbed_funcs"] = tuple(stub_entries)
        if _is_orchestration_callable(record.func):
            metadata["is_orchestration"] = True
        if is_return:
            metadata["is_return"] = True
            metadata["return_vars"] = tuple(outputs)

        node = node_from_callable(node_id, record.func, tuple(inputs), tuple(outputs), metadata)
        self._state.nodes[node_id] = node

    def _allocate_outputs(self, result: Any) -> Tuple[str, ...]:
        values: Tuple[Any, ...]
        if isinstance(result, tuple):
            values = result
        else:
            values = (result,)

        output_vars: List[str] = []
        for value in values:
            if _should_skip_value_tracking(value):
                existing = self._state.next_var()
            else:
                value_id = id(value)
                existing = self._state.value_to_var.get(value_id)
                if existing is None:
                    existing = self._state.next_var()
                    self._state.value_to_var[value_id] = existing
            output_vars.append(existing)
        return tuple(output_vars)

    def _argument_spec(
        self,
        value: Any,
        arg_bindings: Mapping[int, str],
        kw_bindings: Mapping[int, str],
        *,
        keyword: Optional[str] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        value_id = id(value)
        if value_id in self._state.value_to_var:
            var_name = self._state.value_to_var[value_id]
            return var_name, {"kind": "var", "name": var_name}
        if value_id in arg_bindings:
            var_name = arg_bindings[value_id]
            return var_name, {"kind": "arg", "name": var_name}
        if keyword is not None and value_id in kw_bindings:
            var_name = kw_bindings[value_id]
            return var_name, {"kind": "kwarg", "name": var_name}
        if _should_skip_value_tracking(value):
            return None, {"kind": "literal", "value": value}
        return None, {"kind": "value", "value": value}

    def _finalize(self) -> IRGraph:
        graph = IRGraph()
        for node in self._state.nodes.values():
            graph = graph.add_node(node)
        return graph

    def _dispatch_metadata(self, record: OperationRecord):
        func = record.func
        bound_self = getattr(func, "__self__", None)
        if bound_self is None:
            return None
        return getattr(bound_self, "_xcs_dispatch_metadata", None)

    @property
    def _state(self) -> _BuilderState:
        state = getattr(self._state_local, "current", None)
        if state is None:
            raise RuntimeError("IRBuilder state accessed outside of an active trace")
        return state

    @_state.setter
    def _state(self, state: _BuilderState) -> None:
        self._state_local.current = state

    @_state.deleter
    def _state(self) -> None:
        if hasattr(self._state_local, "current"):
            del self._state_local.current


def _is_orchestration_callable(func: Any) -> bool:
    if getattr(func, "_xcs_is_orchestration", False):
        return True
    module = getattr(func, "__module__", "") or ""
    name = getattr(func, "__name__", "") or ""
    lowered = f"{module}.{name}".lower()
    # Internal XCS helpers should not be treated as orchestration.
    safe_prefixes = (
        "ember.xcs.api.",
        "ember.xcs.compiler",
        "ember.xcs.runtime",
        "ember.xcs.utils",
        "jax.",
    )
    if module.startswith(safe_prefixes):
        return False
    orchestration_tokens = (
        "ember",
        "anthropic",
        "openai",
        "llm",
        "model",
        "api",
        "completion",
    )
    return any(token in lowered for token in orchestration_tokens)


_NON_TRACKABLE_TYPES = (int, float, bool, str, bytes)


def _should_skip_value_tracking(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, _NON_TRACKABLE_TYPES)


__all__ = ["IRBuilder"]
