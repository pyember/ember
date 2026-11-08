"""Ember: compose model calls into compound AI systems.

Importing ember reads no files, opens no connections and starts no threads.
Configuration is read the first time a model call needs it.
"""

import importlib.metadata

__all__ = ["__version__"]

__version__ = importlib.metadata.version("ember-ai")
