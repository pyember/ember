"""Convenience re-exports for Ember's accelerated execution utilities.

Most users only need :func:`jit`, :func:`vmap`, and :func:`get_jit_stats`; the
:class:`Config` type remains available for advanced tuning when necessary.

Examples:
    >>> from ember.api.xcs import jit
    >>>
    >>> @jit
    ... def process(data):
    ...     return model(data)
"""

# Re-export the simple API
from ember.xcs import get_jit_stats, jit, vmap

# Config for advanced users (hidden by default)
# Users must explicitly import this
from ember.xcs.config import Config as _Config

# Make Config available but not in __all__
Config = _Config

__all__ = [
    # The 90% API
    "jit",
    "vmap",
    "get_jit_stats",
    # Config is available but not advertised in __all__
]
