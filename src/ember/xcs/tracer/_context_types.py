"""
Context types for XCS tracing.

Defines types that may be used to store additional tracing metadata.
"""

from typing import Any, Dict


class TraceContextData:
    """
    Holds extra context for XCS tracing.

    If you extend XCS, this class can be extended to hold any additional information that the extended tracing
    system might require.
    """

    def __init__(self, extra_info: Dict[str, Any]) -> None:
        self.extra_info = extra_info
