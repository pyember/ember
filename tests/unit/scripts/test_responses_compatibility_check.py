from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_responses_compatibility_harness_passes(tmp_path: Path) -> None:
    script = None
    for candidate in Path(__file__).resolve().parents:
        check_path = candidate / "scripts" / "responses_compatibility_check.py"
        if check_path.exists():
            script = check_path
            break
    assert script is not None, "responses_compatibility_check.py not found"
    result = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
