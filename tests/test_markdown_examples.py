"""Every ```python block in the listed markdown files runs, against a fake model.

Each file's blocks run in order in one namespace, so a later block can use
names an earlier one defined.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ember.llm.request import Message
from ember.operators.verifier import VERIFY_INSTRUCTIONS

ROOT = Path(__file__).resolve().parents[1]
FILES = ("README.md", "docs/operators.md", "docs/non.md")
_BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)


def _reply(model: str, prompt: str | tuple[Message, ...]) -> str:
    if isinstance(prompt, str) and prompt.startswith(VERIFY_INSTRUCTIONS):
        return "The answer is correct.\nVERDICT: PASS"
    return f"an answer from {model}"


@pytest.mark.parametrize("name", FILES)
def test_markdown_examples_run(
    name: str, fake_models: Callable[..., Any], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = fake_models(_reply)
    blocks = _BLOCK.findall((ROOT / name).read_text())
    assert blocks, f"{name} has no python examples"
    namespace: dict[str, object] = {"__name__": "markdown_example"}
    for number, code in enumerate(blocks, start=1):
        exec(compile(code, f"{name} block {number}", "exec"), namespace)
    assert fake.calls, f"{name} never called a model"
    assert capsys.readouterr().out
