"""Every example in the operators and NON docstrings runs and prints what it shows."""

import doctest
import importlib

import pytest

MODULES = [
    "ember.operators.base",
    "ember.operators.model_call",
    "ember.operators.vote",
    "ember.non",
    "ember.non.grammar",
    "ember.non.graph",
]


@pytest.mark.parametrize("name", MODULES)
def test_docstring_examples(name: str) -> None:
    result = doctest.testmod(importlib.import_module(name), raise_on_error=False)
    assert result.attempted > 0, f"{name} has no examples"
    assert result.failed == 0
