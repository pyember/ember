"""Runtime tracer for building execution graphs."""

from __future__ import annotations

import builtins
import sys
import threading
import types
from dataclasses import dataclass, field
from types import FrameType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass(slots=True)
class OperationRecord:
    """A single function invocation observed during tracing."""

    func: Callable[..., Any]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    result: Any
    op_id: int
    dependencies: Tuple[int, ...] = field(default_factory=tuple)


class TracingError(RuntimeError):
    """Raised when tracing fails or recursion is detected."""


@dataclass(slots=True)
class _TracingState:
    """Per-thread tracer bookkeeping."""

    active: bool = False
    operations: List[OperationRecord] = field(default_factory=list)
    result_to_op: Dict[int, int] = field(default_factory=dict)
    counter: int = 0
    target_frame: Optional[FrameType] = None
    frame_dependencies: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    frame_children: Dict[int, List[int]] = field(default_factory=dict)


class PythonTracer:
    """Trace Python execution using ``sys.settrace``."""

    def __init__(self) -> None:
        self._thread_state = threading.local()

    def _get_state(self) -> _TracingState:
        state = getattr(self._thread_state, "state", None)
        if state is None:
            state = _TracingState()
            self._thread_state.state = state
        return state

    def _begin_trace(self, state: _TracingState) -> None:
        state.active = True
        state.operations = []
        state.result_to_op = {}
        state.counter = 0
        state.target_frame = None
        state.frame_dependencies = {}
        state.frame_children = {}

    def _end_trace(self, state: _TracingState) -> None:
        state.active = False
        state.target_frame = None

    def _reset_state(self, state: _TracingState) -> None:
        state.operations = []
        state.result_to_op = {}
        state.counter = 0
        state.target_frame = None
        state.frame_dependencies = {}
        state.frame_children = {}

    def trace_function(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> List[OperationRecord]:
        """Return a list of operations recorded while executing `func`."""
        state = self._get_state()
        if state.active:
            raise TracingError("PythonTracer is already tracing a function in this thread")
        self._begin_trace(state)

        previous = sys.gettrace()
        sys.settrace(self._trace)
        try:
            result = func(*args, **kwargs)
        except Exception:  # pragma: no cover
            self._reset_state(state)
            raise
        finally:
            sys.settrace(previous)
            self._end_trace(state)

        main_record = OperationRecord(
            func=func,
            args=args,
            kwargs=kwargs,
            result=result,
            op_id=state.counter,
            dependencies=tuple(range(state.counter)),
        )
        records = list(state.operations)
        records.append(main_record)
        self._reset_state(state)
        return records

    # Internal helpers --------------------------------------------------
    def _trace(
        self, frame: FrameType, event: str, arg: Any
    ) -> Optional[Callable[..., Any]]:
        state = self._get_state()

        if not state.active:
            return None

        if state.target_frame is None and event == "call":
            state.target_frame = frame

        if not self._is_in_target_stack(frame):
            return None

        if event == "call":
            if self._is_tracer_frame(frame):
                return None
            frame_id = id(frame)
            state.frame_dependencies[frame_id] = tuple(self._extract_dependencies(frame))
            state.frame_children[frame_id] = []
        elif event == "return":
            if frame is state.target_frame:
                return self._trace
            self._record_operation(frame, arg)
        return self._trace

    def _is_in_target_stack(self, frame: FrameType) -> bool:
        state = self._get_state()
        current: Optional[FrameType] = frame
        while current is not None:
            if current is state.target_frame:
                return True
            current = current.f_back
        return False

    def _is_tracer_frame(self, frame: FrameType) -> bool:
        return frame.f_code.co_filename == __file__

    def _extract_dependencies(self, frame: FrameType) -> Sequence[int]:
        state = self._get_state()
        deps: List[int] = []
        for value in frame.f_locals.values():
            op_id = state.result_to_op.get(id(value))
            if op_id is not None:
                deps.append(op_id)
        return deps

    def _record_operation(self, frame: FrameType, result: Any) -> None:
        state = self._get_state()
        func = self._resolve_callable(frame)
        if func is None:
            return

        frame_id = id(frame)
        arg_deps = state.frame_dependencies.pop(frame_id, tuple())
        child_deps = state.frame_children.pop(frame_id, [])
        dependencies = _dedupe_dependencies(arg_deps, child_deps)

        args, kwargs = self._extract_arguments(frame)
        record = OperationRecord(
            func=func,
            args=args,
            kwargs=kwargs,
            result=result,
            op_id=state.counter,
            dependencies=dependencies,
        )
        state.operations.append(record)
        if _should_track_result(result):
            state.result_to_op[id(result)] = state.counter
        op_id = state.counter
        state.counter += 1

        parent = frame.f_back
        if parent is None or not self._is_in_target_stack(parent) or self._is_tracer_frame(parent):
            return
        parent_children = state.frame_children.setdefault(id(parent), [])
        parent_children.append(op_id)

    def _resolve_callable(self, frame: FrameType) -> Optional[Callable[..., Any]]:
        name = frame.f_code.co_name

        candidate = frame.f_globals.get(name)
        if callable(candidate):
            return candidate

        enclosing = frame.f_back
        while enclosing is not None:
            candidate = enclosing.f_locals.get(name)
            if callable(candidate):
                return candidate
            enclosing = enclosing.f_back

        self_obj = frame.f_locals.get("self")
        if self_obj is not None:
            attr = getattr(self_obj, name, None)
            if callable(attr):
                return attr

        built = getattr(builtins, name, None)
        if callable(built):
            return built

        if name != "<lambda>":
            return None

        closure_cells: Optional[Tuple[types.CellType, ...]] = None
        freevars = frame.f_code.co_freevars
        if freevars:
            closure_list: list[types.CellType] = []
            for var in freevars:
                value = frame.f_locals.get(var, frame.f_globals.get(var))
                closure_list.append(_make_cell(value))
            closure_cells = tuple(closure_list)

        return types.FunctionType(
            frame.f_code,
            frame.f_globals,
            name,
            None,
            closure_cells,
        )

    def _extract_arguments(self, frame: FrameType) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        code = frame.f_code
        positional: List[Any] = []
        for _index, varname in enumerate(code.co_varnames[: code.co_argcount]):
            if varname == "self":
                continue
            positional.append(frame.f_locals.get(varname))
        keywords: Dict[str, Any] = {}
        start = code.co_argcount
        end = start + code.co_kwonlyargcount
        for varname in code.co_varnames[start:end]:
            if varname in frame.f_locals:
                keywords[varname] = frame.f_locals[varname]
        return tuple(positional), keywords


_NON_TRACKABLE_TYPES = (int, float, bool, str, bytes)


def _should_track_result(value: Any) -> bool:
    if value is None:
        return False
    return not isinstance(value, _NON_TRACKABLE_TYPES)


def _make_cell(value: Any) -> types.CellType:
    # Helper borrowed from CPython tests to create cell objects on demand.
    def capture() -> Any:
        return value

    closure = capture.__closure__
    if closure is None:
        raise RuntimeError("Failed to synthesize closure cell for tracer")
    return closure[0]


def _dedupe_dependencies(arg_deps: Sequence[int], child_deps: Sequence[int]) -> Tuple[int, ...]:
    ordered: List[int] = []
    seen: set[int] = set()
    for dep in tuple(arg_deps) + tuple(child_deps):
        if dep in seen:
            continue
        seen.add(dep)
        ordered.append(dep)
    return tuple(ordered)


__all__ = ["PythonTracer", "TracingError", "OperationRecord"]
