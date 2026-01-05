"""Backward-compatible tracer alias."""

from ember.xcs.tracing.python_tracer import OperationRecord, PythonTracer, TracingError

__all__ = ["OperationRecord", "PythonTracer", "TracingError"]
