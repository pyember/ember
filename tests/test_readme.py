"""The README's Operators and NON examples run, against the fake model call."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ember.operators.verifier import VERIFY_INSTRUCTIONS

README = Path(__file__).resolve().parents[1] / "README.md"
SECTIONS = ("## NON: Compound AI in One Line", "### 2. Operators — Composable Building Blocks")


def _section(text: str, heading: str) -> str:
    start = text.index(heading + "\n") + len(heading)
    following = re.search(r"^#{2,3} ", text[start:], flags=re.MULTILINE)
    return text[start : start + following.start()] if following else text[start:]


def _blocks() -> list[tuple[str, str]]:
    text = README.read_text()
    found = []
    for heading in SECTIONS:
        for number, code in enumerate(
            re.findall(r"```python\n(.*?)```", _section(text, heading), re.DOTALL)
        ):
            found.append((f"{heading} #{number}", code))
    return found


def _reply(model: str, prompt: str) -> str:
    if prompt.startswith(VERIFY_INSTRUCTIONS):
        return "Consistent with the question.\nVERDICT: PASS"
    return f"{model} answers"


@pytest.mark.parametrize(("name", "code"), _blocks(), ids=[name for name, _ in _blocks()])
def test_readme_block_runs(
    name: str, code: str, fake_models: Callable[..., Any], capsys: pytest.CaptureFixture[str]
) -> None:
    fake_models(_reply)
    exec(compile(code, f"README.md: {name}", "exec"), {"__name__": "readme"})
    assert capsys.readouterr().out


def test_both_sections_have_examples() -> None:
    names = [name for name, _ in _blocks()]
    assert all(any(name.startswith(heading) for name in names) for heading in SECTIONS)
