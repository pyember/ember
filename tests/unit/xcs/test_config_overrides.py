"""Tests for Config.apply_overrides convenience method."""

from __future__ import annotations

import pytest

from ember.xcs.config import Config
from ember.xcs.errors import XCSError


def test_apply_overrides_returns_new_config() -> None:
    base = Config(parallel=True, cache=True, profile=False, max_workers=None, max_memory_mb=None)
    updated = base.apply_overrides({"parallel": False, "max_workers": 8})
    assert updated.parallel is False
    assert updated.max_workers == 8
    # Original instance remains unchanged
    assert base.parallel is True
    assert base.max_workers is None


def test_apply_overrides_rejects_unknown_keys() -> None:
    base = Config()
    with pytest.raises(XCSError):
        base.apply_overrides({"unknown": True})
