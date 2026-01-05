"""Top-level package for the Ember compound AI framework.

Core APIs are available directly from the package:

    import ember

    # Invoke models
    ember.models("gpt-4", "Hello")
    ember.models.list()

    # Define operators
    @ember.op
    def pipeline(text: str) -> str:
        return ember.models("gpt-4", text)

    # Load data
    for record in ember.stream("mmlu", max_items=10):
        print(record.question)

    data = ember.load("mmlu", max_items=100)

Specialized APIs remain accessible via submodules:
    - ``ember.api.eval``: Evaluation pipelines
    - ``ember.api.xcs``: JIT compilation (``jit``, ``vmap``)
    - ``ember.non``: Experimental NON graph toolkit
"""

from __future__ import annotations

import importlib.metadata

from ember.api.data import load, stream
from ember.api.decorators import op
from ember.api.models import models

try:
    __version__ = importlib.metadata.version("ember-ai")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__", "load", "models", "op", "stream"]
