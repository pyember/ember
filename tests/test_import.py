"""`import ember` reads no config, starts no threads and loads no heavy dependency."""

import subprocess
import sys
import textwrap
from pathlib import Path


def test_import_has_no_side_effects(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".ember").mkdir(parents=True)
    (home / ".ember" / "config.yaml").write_text("this: is: not: yaml: [\n")
    script = textwrap.dedent(
        """
        import sys, threading
        before = threading.active_count()
        import ember, ember.llm, ember.context, ember.config, ember.errors, ember.xcs
        import ember.data, ember.eval, ember.cli, ember.operators, ember.non
        assert threading.active_count() == before, "import started a thread"
        heavy = {"datasets", "huggingface_hub", "pyarrow", "jax", "equinox", "torch"}
        heavy |= {"transformers", "openai", "anthropic"}
        loaded = heavy & set(sys.modules)
        assert not loaded, f"import loaded {sorted(loaded)}"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
