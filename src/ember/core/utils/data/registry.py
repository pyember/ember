"""Data registry for dataset management.

This lightweight registry provides a thread-safe place to register and
retrieve named data sources. It currently delegates loading to the public
`ember.api.data` API and exists to keep a stable abstraction boundary inside
the core utilities layer.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ember._internal.context import EmberContext


class DataRegistry:
    """Registry for dataset loaders.

    The registry stores named sources and exposes simple helpers to load and
    register them. Access is guarded by a `threading.Lock` for thread safety.
    """

    def __init__(self, context: Optional[EmberContext] = None):
        """Initialize the registry.

        Args:
            context: Optional `EmberContext` for future configuration needs.
        """
        self._context = context
        self._sources: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def load(self, name: str, **kwargs) -> Any:
        """Load a dataset by name via the public data API.

        This is a thin wrapper around `ember.api.data.stream()`.

        Args:
            name: Dataset name.
            **kwargs: Keyword arguments forwarded to `data.stream()`.

        Returns:
            Stream iterator over dataset items.
        """
        # Use the data API directly for now.
        from ember.api import data

        return data.stream(name, **kwargs)

    def register(self, name: str, source: Any) -> None:
        """Register a data source under a name.

        Args:
            name: Unique source name.
            source: Data source implementation.
        """
        with self._lock:
            self._sources[name] = source
