# trace_context.py

from typing import List, Dict, Any, Optional
import contextvars

# A thread-local variable that stores the active TraceContext, if any
_current_trace_context = contextvars.ContextVar("current_trace_context", default=None)

class TraceContext:
    """
    Holds a list of 'TraceRecord' objects describing each operator's
    input-output. If active, operators will log to this context.
    """

    def __init__(self):
        self.records: List["TraceRecord"] = []

    def add_record(self, record: "TraceRecord"):
        self.records.append(record)

    def get_records(self) -> List["TraceRecord"]:
        return self.records

class TraceRecord:
    """
    Each record logs:
      - operator_name
      - operator_class
      - input_data (raw)
      - output_data (raw)
      - optional custom metadata
    """
    def __init__(
        self,
        operator_name: str,
        operator_class: str,
        input_data: Any,
        output_data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.operator_name = operator_name
        self.operator_class = operator_class
        self.input_data = input_data
        self.output_data = output_data
        self.metadata = metadata or {}

class RecordingContext:
    """
    A context manager that sets up a new TraceContext, which operators can write into.
    """

    def __enter__(self):
        new_context = TraceContext()
        self._token = _current_trace_context.set(new_context)
        return new_context

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Revert to previous context
        _current_trace_context.reset(self._token)

def get_current_trace_context() -> Optional[TraceContext]:
    return _current_trace_context.get()