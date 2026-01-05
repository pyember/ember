"""Tests for orchestration transform failure signalling."""

from __future__ import annotations

import threading

import pytest

from ember.xcs.api.transforms import _execute_orchestration_batch
from ember.xcs.config import Config


def test_parallel_failure_surfaces() -> None:
    def flaky(value: int) -> int:
        if threading.current_thread().name.startswith("ThreadPoolExecutor"):
            raise ValueError("fields were not initialised: synthetic test")
        return value * 2

    args = ([1, 2],)
    kwargs = {}
    config = Config(max_workers=4)

    with pytest.raises(ValueError, match="fields were not initialised"):
        _execute_orchestration_batch(
            flaky,
            args,
            kwargs,
            in_axes=0,
            out_axes=0,
            axis_size=None,
            config=config,
        )
