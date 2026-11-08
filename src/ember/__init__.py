"""Ember: compose model calls into compound AI systems.

Importing ember reads no files, opens no connections and starts no threads.
Configuration is read the first time a model call needs it.
"""

import importlib.metadata

from ember.llm.api import models
from ember.non import build_graph
from ember.operators import op

__all__ = ["__version__", "build_graph", "models", "op"]

__version__ = importlib.metadata.version("ember-ai")
