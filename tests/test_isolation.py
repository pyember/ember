"""The suite never sees the real home directory, config file or provider keys."""

import os
from pathlib import Path

from ember import config


def test_tests_cannot_reach_the_real_home_or_config(real_home: Path) -> None:
    assert Path.home() != real_home
    # Not `real_home` itself: on Windows the temporary directory is inside it.
    assert real_home / ".ember" not in config.config_path().parents


def test_tests_run_without_provider_keys() -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        assert name not in os.environ
