from __future__ import annotations

import logging
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

from pydantic import model_validator

from ember.core.exceptions import (
    InvalidArgumentError,
    InvalidPromptError,
    SpecificationValidationError,
)
from ember.core.types import EmberModel

logger = logging.getLogger(__name__)

InputModelT = TypeVar("InputModelT", bound=EmberModel)
OutputModelT = TypeVar("OutputModelT", bound=EmberModel)


class Specification(EmberModel, Generic[InputModelT, OutputModelT]):
    """Base class representing an operator's specification.

    Attributes:
        prompt_template (Optional[str]): Template string that may reference input field names.
        structured_output (Optional[Type[OutputModelT]]): Pydantic model class used for output validation.
        input_model (Optional[Type[InputModelT]]): Pydantic model class defining the expected input fields.
        check_all_placeholders (bool): Flag to enforce that all required placeholders are included in the prompt template.
    """

    prompt_template: Optional[str] = None
    structured_output: Optional[Type[OutputModelT]] = None
    input_model: Optional[Type[InputModelT]] = None
    check_all_placeholders: bool = True

    def _get_required_fields(self) -> List[str]:
        """Retrieve the names of required fields from the input model.

        Returns:
            List[str]: A list of required field names if an input_model is defined; otherwise, an empty list.
        """
        if self.input_model is None:
            return []
        return [
            field_name
            for field_name, field in self.input_model.model_fields.items()
            if field.is_required()
        ]

    @model_validator(mode="after")
    def _validate_template(self) -> Specification[InputModelT, OutputModelT]:
        """Ensure the prompt template contains all required placeholders.

        Returns:
            Specification[InputModelT, OutputModelT]: The validated specification instance.

        Raises:
            InvalidPromptError: If one or more required placeholders are missing in the prompt template.
        """
        if (
            self.prompt_template is not None
            and self.input_model is not None
            and self.check_all_placeholders
        ):
            required_fields: List[str] = self._get_required_fields()
            missing_fields: List[str] = [
                field
                for field in required_fields
                if f"{{{field}}}" not in self.prompt_template
            ]
            if missing_fields:
                missing_placeholder = ", ".join(missing_fields)
                error_msg: str = (
                    f"Missing placeholders in prompt_template: {missing_placeholder}"
                )
                logger.error(error_msg)
                raise InvalidPromptError.with_context(
                    error_msg,
                    missing_placeholders=missing_fields,
                    required_fields=required_fields,
                    template=self.prompt_template,
                )
        return self

    def render_prompt(self, *, inputs: Union[Dict[str, Any], EmberModel]) -> str:
        """Render a prompt using the provided inputs.

        If a prompt_template is specified, formats it using the given inputs.
        Otherwise, if an input_model is defined, concatenates its required fields' values.
        If neither is available, raises an error.

        Args:
            inputs (Union[Dict[str, Any], EmberModel]): Input values as a dictionary or model.

        Returns:
            str: The rendered prompt string.

        Raises:
            InvalidPromptError: If placeholder validation is enabled without an input_model,
                if a required placeholder value is missing, or if neither a prompt_template nor an input_model is defined.
        """
        if self.check_all_placeholders and self.input_model is None:
            error_msg: str = "Missing input_model for placeholder validation."
            logger.error(error_msg)
            raise InvalidPromptError.with_context(
                error_msg,
                validation_enabled=self.check_all_placeholders,
                input_model=None,
            )

        # Convert inputs to dictionary if it's a model
        input_dict: Dict[str, Any] = inputs
        if isinstance(inputs, EmberModel):
            input_dict = inputs.as_dict()

        if self.prompt_template is not None:
            try:
                prompt: str = self.prompt_template.format(**input_dict)
                return prompt
            except KeyError as key_err:
                missing_key = str(key_err)
                error_msg: str = f"Missing input for placeholder: {missing_key}"
                logger.error(error_msg)
                raise InvalidPromptError.with_context(
                    error_msg,
                    missing_placeholder=missing_key,
                    available_keys=list(input_dict.keys()),
                    template=self.prompt_template,
                ) from key_err

        if self.input_model is not None:
            required_fields: List[str] = self._get_required_fields()
            return "\n".join(
                str(input_dict.get(field, "")) for field in required_fields
            )

        error_msg: str = (
            "No prompt_template or input_model defined for rendering prompt."
        )
        logger.error(error_msg)
        raise InvalidPromptError.with_context(
            error_msg,
            prompt_template=self.prompt_template,
            input_model=self.input_model,
        )

    def model_json_schema(self, *, by_alias: bool = True) -> Dict[str, Any]:
        """Return the JSON schema for the input model.

        Args:
            by_alias (bool): Whether to use field aliases in the schema.

        Returns:
            Dict[str, Any]: The JSON schema for the input_model or an empty dict if it is not defined.
        """
        if self.input_model is not None:
            return self.input_model.model_json_schema(by_alias=by_alias)
        return {}

    def _validate_data(
        self,
        *,
        data: Union[Dict[str, Any], EmberModel],
        model: Type[EmberModel],
        model_label: str,
    ) -> EmberModel:
        """Validate the provided data against a specified Pydantic model.

        Args:
            data (Union[Dict[str, Any], EmberModel]): The data to validate.
            model (Type[EmberModel]): The Pydantic model for validation.
            model_label (str): A label for error messages (e.g., "Input" or "Output").

        Returns:
            EmberModel: A validated instance of the specified model.

        Raises:
            SpecificationValidationError: If a EmberModel instance does not match the expected model.
            InvalidArgumentError: If the data is neither a dict nor a Pydantic model.
        """
        if isinstance(data, dict):
            return model.model_validate(data)
        if isinstance(data, EmberModel):
            if not isinstance(data, model):
                error_msg: str = f"{model_label} model mismatch. Expected {model.__name__}, got {type(data).__name__}."
                logger.error(error_msg)
                raise SpecificationValidationError.with_context(
                    error_msg,
                    error_type="model_mismatch",
                    expected_model=model.__name__,
                    actual_model=type(data).__name__,
                    model_label=model_label,
                )
            return data

        error_msg: str = (
            f"{model_label} must be a dict or an EmberModel, got {type(data).__name__}."
        )
        logger.error(error_msg)
        raise InvalidArgumentError.with_context(
            error_msg,
            error_type="invalid_input_type",
            expected_types=["dict", "EmberModel"],
            actual_type=type(data).__name__,
            model_label=model_label,
        )

    def validate_inputs(
        self, *, inputs: Union[Dict[str, Any], InputModelT]
    ) -> Union[InputModelT, Dict[str, Any]]:
        """Validate and parse raw inputs per the defined input model.

        Args:
            inputs (Union[Dict[str, Any], InputModelT]): Raw input data as a dictionary or Pydantic model.

        Returns:
            Union[InputModelT, Dict[str, Any]]: A validated input model instance or the original inputs.

        Raises:
            SpecificationValidationError: If the provided BaseModel does not match the expected input_model.
            InvalidArgumentError: If inputs are neither a dict nor a Pydantic model.
        """
        if self.input_model is not None:
            return self._validate_data(
                data=inputs, model=self.input_model, model_label="Input"
            )
        if isinstance(inputs, (dict, EmberModel)):
            return inputs

        error_msg: str = (
            f"Inputs must be a dict or an EmberModel, got {type(inputs).__name__}."
        )
        logger.error(error_msg)
        raise InvalidArgumentError.with_context(
            error_msg,
            error_type="invalid_input_type",
            expected_types=["dict", "EmberModel"],
            actual_type=type(inputs).__name__,
            model_available=self.input_model is not None,
        )

    def validate_output(
        self, *, output: Union[Dict[str, Any], OutputModelT]
    ) -> Union[OutputModelT, Dict[str, Any]]:
        """Validate and parse raw output using the structured output model.

        Args:
            output (Union[Dict[str, Any], OutputModelT]): Raw output data as a dictionary or Pydantic model.

        Returns:
            Union[OutputModelT, Dict[str, Any]]: A validated structured output instance or the original output.

        Raises:
            SpecificationValidationError: If the provided BaseModel does not match the expected structured_output.
            InvalidArgumentError: If output is neither a dict nor a Pydantic model when a structured_output model is defined.
        """
        if self.structured_output is not None:
            return self._validate_data(
                data=output, model=self.structured_output, model_label="Output"
            )
        return output
