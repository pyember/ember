"""Helpers for testing example scripts run without errors."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


class ExampleTest:
    """Base class for example tests that verifies examples run successfully."""

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def examples_root(self) -> Path:
        return self.repo_root / "examples"

    def run_example(
        self, example_path: str, *, timeout: float
    ) -> tuple[str, str, float, int]:
        """Run an example script and capture its output.

        Args:
            example_path: Path relative to examples/ directory.
            timeout: Maximum seconds to allow for execution.

        Returns:
            Tuple of (stdout, stderr, duration_seconds, returncode).

        Raises:
            FileNotFoundError: If the example script does not exist.
            TimeoutError: If execution exceeds the timeout.
        """
        full_path = self.examples_root / example_path
        if not full_path.exists():
            raise FileNotFoundError(f"Example not found: {full_path}")

        env = os.environ.copy()
        project_root = str(self.repo_root)
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            os.pathsep.join([project_root, existing_pythonpath])
            if existing_pythonpath
            else project_root
        )

        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, str(full_path)],
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Example exceeded timeout ({timeout:.2f}s): {example_path}"
            ) from exc
        duration = time.time() - start
        return result.stdout, result.stderr, duration, result.returncode

    def run_example_test(self, example_path: str, *, timeout: float = 30.0) -> None:
        """Run an example and verify it completes successfully.

        Args:
            example_path: Path relative to examples/ directory.
            timeout: Maximum seconds to allow for execution.
        """
        stdout, stderr, duration, returncode = self.run_example(
            example_path, timeout=timeout
        )

        if returncode != 0:
            pytest.fail(
                f"Example failed with exit code {returncode}.\n"
                f"Stderr:\n{stderr}\n"
                f"Stdout:\n{stdout}"
            )


# Backwards compatibility alias
ExampleGoldenTest = ExampleTest
