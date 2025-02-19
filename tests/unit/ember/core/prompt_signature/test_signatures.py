# File: tests/test_signatures.py
"""
Tests for prompt signature functionality.

This file verifies:
  - Correct prompt rendering with valid inputs.
  - Input validation via both dict and model.
  - Output validation from raw dicts.
  - Proper error raising for missing placeholders or misconfigured signatures.
"""

from typing import Dict, Type
import pytest
from pydantic import BaseModel
from src.ember.core.registry.prompt_signature.signatures import Signature


class DummyInput(BaseModel):
    name: str


class DummyOutput(BaseModel):
    result: str


class DummySignature(Signature):
    prompt_template: str = "Hello, {name}!"
    input_model: Type[DummyInput] = DummyInput
    structured_output: Type[DummyOutput] = DummyOutput
    check_all_placeholders: bool = True


def test_render_prompt_valid() -> None:
    sig = DummySignature()
    prompt: str = sig.render_prompt({"name": "Test"})
    assert prompt == "Hello, Test!"


def test_render_prompt_missing_placeholder() -> None:
    # When check_all_placeholders is enabled, a missing placeholder should raise a ValueError.
    class NoPlaceholderSignature(Signature):
        prompt_template: str = "Hello!"  # Missing {name}
        input_model: Type[DummyInput] = DummyInput
        check_all_placeholders: bool = True

    with pytest.raises(ValueError) as excinfo:
        NoPlaceholderSignature()
    assert "Required input 'name'" in str(excinfo.value)


def test_validate_inputs_with_dict() -> None:
    sig = DummySignature()
    valid_input: Dict[str, str] = {"name": "Alice"}
    validated = sig.validate_inputs(valid_input)
    assert isinstance(validated, DummyInput)
    assert validated.name == "Alice"


def test_validate_inputs_with_model() -> None:
    sig = DummySignature()
    model_instance = DummyInput(name="Bob")
    validated = sig.validate_inputs(model_instance)
    assert isinstance(validated, DummyInput)
    assert validated.name == "Bob"


def test_validate_inputs_invalid_type() -> None:
    sig = DummySignature()
    with pytest.raises(TypeError):
        sig.validate_inputs("not a dict or model")


def test_validate_output() -> None:
    sig = DummySignature()
    raw_output: Dict[str, str] = {"result": "Success"}
    validated = sig.validate_output(raw_output)
    assert isinstance(validated, DummyOutput)
    assert validated.result == "Success"


def test_misconfigured_signature_missing_input_model() -> None:
    # When no input_model is defined, rendering should raise a ValueError.
    class MisconfiguredSignature(Signature):
        prompt_template: str = "Hi, {name}!"
        check_all_placeholders: bool = True

    sig = MisconfiguredSignature()
    with pytest.raises(ValueError):
        sig.render_prompt({"name": "Test"})


def test_misconfigured_signature_incompatible_model() -> None:
    # Test that validate_inputs raises an error when given an instance of a wrong model.
    class AnotherInput(BaseModel):
        other: str

    class IncompatibleSignature(Signature):
        prompt_template: str = "Hi, {other}!"
        input_model: Type[AnotherInput] = AnotherInput
        check_all_placeholders: bool = True

    sig = IncompatibleSignature()
    wrong_instance = DummyInput(name="Test")
    with pytest.raises(ValueError) as excinfo:
        sig.validate_inputs(wrong_instance)
    assert "Input model mismatch" in str(excinfo.value)