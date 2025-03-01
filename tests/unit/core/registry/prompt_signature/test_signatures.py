from __future__ import annotations
from typing import Dict, Type

import pytest
from pydantic import BaseModel

from ember.core.registry.prompt_signature.exceptions import (
    InvalidInputTypeError,
    MismatchedModelError,
    PlaceholderMissingError,
)
from ember.core.registry.prompt_signature.signatures import Signature


class DummyInput(BaseModel):
    """Dummy input model for testing prompt signatures.

    Attributes:
        name (str): The name used for prompt generation.
    """

    name: str


class DummyOutput(BaseModel):
    """Dummy output model for testing prompt signature functionality.

    Attributes:
        result (str): The result produced by the prompt.
    """

    result: str


class DummySignature(Signature[DummyInput, DummyOutput]):
    """Dummy signature for testing prompt rendering, input validation, and output validation.

    Attributes:
        prompt_template (str): Template string for greeting using a name.
        input_model (Type[DummyInput]): Model used for input validation.
        structured_output (Type[DummyOutput]): Model used for output validation.
        check_all_placeholders (bool): Flag to enforce that all required placeholders are present.
    """

    prompt_template: str = "Hello, {name}!"
    input_model: Type[DummyInput] = DummyInput
    structured_output: Type[DummyOutput] = DummyOutput
    check_all_placeholders: bool = True


def test_render_prompt_valid() -> None:
    """Test that render_prompt produces the correct output for valid input."""
    dummy_signature: DummySignature = DummySignature()
    rendered_prompt: str = dummy_signature.render_prompt(inputs={"name": "Test"})
    assert rendered_prompt == "Hello, Test!"


def test_render_prompt_missing_placeholder() -> None:
    """Test that instantiation fails when a required placeholder is missing in the prompt template."""

    class NoPlaceholderSignature(Signature[DummyInput, DummyOutput]):
        """Signature missing a required placeholder in its prompt template."""

        prompt_template: str = "Hello!"  # Missing the '{name}' placeholder.
        input_model: Type[DummyInput] = DummyInput
        check_all_placeholders: bool = True

    with pytest.raises(PlaceholderMissingError) as exc_info:
        _ = NoPlaceholderSignature()  # Validation is triggered upon instantiation.
    assert "name" in str(exc_info.value)


def test_validate_inputs_with_dict() -> None:
    """Test that validate_inputs correctly parses a dictionary input."""
    dummy_signature: DummySignature = DummySignature()
    input_data: Dict[str, str] = {"name": "Alice"}
    validated_input = dummy_signature.validate_inputs(inputs=input_data)
    assert isinstance(validated_input, DummyInput)
    assert validated_input.name == "Alice"


def test_validate_inputs_with_model() -> None:
    """Test that validate_inputs accepts an already valid Pydantic model."""
    dummy_signature: DummySignature = DummySignature()
    input_instance: DummyInput = DummyInput(name="Bob")
    validated_input = dummy_signature.validate_inputs(inputs=input_instance)
    assert isinstance(validated_input, DummyInput)
    assert validated_input.name == "Bob"


def test_validate_inputs_invalid_type() -> None:
    """Test that validate_inputs raises an error when given an invalid input type."""
    dummy_signature: DummySignature = DummySignature()
    with pytest.raises(InvalidInputTypeError):
        dummy_signature.validate_inputs(inputs="invalid input type")


def test_validate_output() -> None:
    """Test that validate_output correctly parses dictionary output data."""
    dummy_signature: DummySignature = DummySignature()
    output_data: Dict[str, str] = {"result": "Success"}
    validated_output = dummy_signature.validate_output(output=output_data)
    assert isinstance(validated_output, DummyOutput)
    assert validated_output.result == "Success"


def test_misconfigured_signature_missing_input_model() -> None:
    """Test that rendering a prompt raises an error when the input_model is missing."""

    class MisconfiguredSignature(Signature):
        """Signature configured without an input_model."""

        prompt_template: str = "Hi, {name}!"
        check_all_placeholders: bool = True

    misconfigured_signature = MisconfiguredSignature()
    with pytest.raises(PlaceholderMissingError):
        misconfigured_signature.render_prompt(inputs={"name": "Test"})


def test_misconfigured_signature_incompatible_model() -> None:
    """Test that validate_inputs raises an error when the provided input model type is incompatible."""

    class AnotherInput(BaseModel):
        """Alternate input model for testing signature compatibility."""

        other: str

    class IncompatibleSignature(Signature[AnotherInput, DummyOutput]):
        """Signature expecting a different input model type."""

        prompt_template: str = "Hi, {other}!"
        input_model: Type[AnotherInput] = AnotherInput
        check_all_placeholders: bool = True

    incompatible_signature = IncompatibleSignature()
    wrong_input_instance: DummyInput = DummyInput(name="Test")
    with pytest.raises(MismatchedModelError) as exc_info:
        incompatible_signature.validate_inputs(inputs=wrong_input_instance)
    assert "Input model mismatch" in str(exc_info.value)


def test_render_prompt_with_no_template_but_input_model() -> None:
    """Test that render_prompt falls back to the input_model value when no prompt_template is provided."""

    class NoTemplateSignature(Signature[DummyInput, DummyOutput]):
        """Signature without a prompt_template but with an input model."""

        input_model: Type[DummyInput] = DummyInput
        check_all_placeholders: bool = False

    no_template_signature = NoTemplateSignature()
    rendered_prompt: str = no_template_signature.render_prompt(inputs={"name": "Test"})
    assert "Test" in rendered_prompt


def test_render_prompt_no_template_no_input_model() -> None:
    """Test that render_prompt raises an error when neither prompt_template nor input_model is provided."""

    class EmptySignature(Signature):
        """Empty signature with no prompt_template or input_model."""

        check_all_placeholders: bool = False

    empty_signature = EmptySignature()
    with pytest.raises(PlaceholderMissingError):
        empty_signature.render_prompt(inputs={})
