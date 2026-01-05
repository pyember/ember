"""XCS Engine should respect Config.max_workers for parallel execution.

This test constructs a trivial function with independent branches so that
the engine will choose the parallel path, and verifies that execution completes
without error when a small max_workers is provided.

Note: This is a behavioral smoke test; precise worker count introspection is
left to integration tests due to executor abstraction.
"""

from __future__ import annotations

from ember.xcs import jit
from ember.xcs.config import Config


def independent(a: int) -> int:
    return a + 1


@jit
def multi_branch(x: int) -> int:
    # Independent branches; XCS IR should parallelize
    r1 = independent(x)
    r2 = independent(x)
    r3 = independent(x)
    return r1 + r2 + r3


def test_engine_parallel_respects_max_workers():
    cfg = Config(max_workers=2)

    # Call through the jitted function with explicit config override.
    # Our simple jit front-door takes config via keyword.
    result = multi_branch(10, _config=cfg)  # type: ignore[arg-type]
    assert result == (11 + 11 + 11)
