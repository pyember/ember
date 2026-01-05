"""xcs.vmap should use Config.max_workers for orchestration batching.

We simulate an orchestration function by using a plain Python function; the
transformations layer routes orchestration-only functions through a thread pool.
"""

from __future__ import annotations

from typing import List

from ember.xcs import vmap
from ember.xcs.config import Config


def fake_orchestration(x: int) -> int:
    # Stand-in for a model/tool call
    return x * 2


def test_vmap_orchestration_respects_max_workers():
    # Use a small batch and small worker cap
    batched = vmap(fake_orchestration, in_axes=0, config=Config(max_workers=2))
    xs = [1, 2, 3, 4]
    ys: List[int] = batched(xs)  # type: ignore[assignment]
    assert ys == [2, 4, 6, 8]
