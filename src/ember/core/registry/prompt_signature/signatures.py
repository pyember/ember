from __future__ import annotations
from typing import Dict, Optional, Type, Union, TypeVar, Generic

from pydantic import BaseModel, model_validator

InputModelT = TypeVar("InputModelT", bound=BaseModel)
OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Signature(BaseModel, Generic[InputModelT, OutputModelT]):
    """Base class representing an operator's signature.

    Attributes:
        prompt_template (Optional[str]): Template string that may reference input names.
        structured_output (Optional[Type[BaseModel]]): Pydantic model class for validating the output.
        input_model (Optional[Type[BaseModel]]): Pydantic model class defining the expected input fields.
        check_all_placeholders (bool): Flag to enforce the inclusion of all required placeholders.
    """

    prompt_template: Optional[str] = None
    structured_output: Optional[Type[OutputModelT]] = None
    input_model: Optional[Type[InputModelT]] = None
    check_all_placeholders: bool = False

    @model_validator(mode="after")
    def check_template(self) -> Signature[InputModelT, OutputModelT]:
        """Validates that the prompt_template contains required input placeholders."""
        if (
            self.prompt_template is not None
            and self.input_model is not None
            and self.check_all_placeholders
        ):
            required_fields = [
                name
                for name, field in self.input_model.model_fields.items()
                if field.is_required()
            ]
            for field_name in required_fields:
                placeholder: str = f"{{{field_name}}}"
                if placeholder not in self.prompt_template:
                    raise ValueError(
                        f"Required input '{field_name}' not found in prompt_template."
                    )
        return self

    def render_prompt(self, inputs: Dict[str, object]) -> str:
        """Renders a prompt string based on provided inputs and configuration.

        If a prompt_template is defined, the method formats it using the values from 'inputs'.
        Otherwise, if an input_model is available, it concatenates the values of the required fields.
        If neither is available, a ValueError is raised.

        Args:
            inputs (Dict[str, object]): Dictionary containing input values.

        Returns:
            str: The prompt string generated from the inputs.

        Raises:
            ValueError: If neither a prompt_template nor an input_model is defined for rendering prompt.
        """
        if self.check_all_placeholders and self.input_model is None:
            raise ValueError("Missing input_model for placeholder validation.")
        if self.prompt_template:
            return self.prompt_template.format(**inputs)
        if self.input_model:
            required_fields = [
                name
                for name, field in self.input_model.model_fields.items()
                if getattr(field, "required", False)
            ]
            return "\n".join(str(inputs.get(name, "")) for name in required_fields)
        raise ValueError(
            "No prompt_template or input_model defined for rendering prompt."
        )

    def model_json_schema(self) -> Dict[str, object]:
        """Retrieves the JSON schema for the input model.

        If an input_model is defined, its JSON schema is returned; otherwise, an empty dictionary is provided.

        Returns:
            Dict[str, object]: The JSON schema of the input_model or an empty dict if not defined.
        """
        if self.input_model is not None:
            return self.input_model.model_json_schema()
        return {}

    def validate_inputs(
        self, inputs: Union[Dict[str, object], InputModelT]
    ) -> Union[InputModelT, Dict[str, object]]:
        """Validates and parses raw inputs using the defined input model.

        When an input_model is specified, this method converts the raw input (either a dict or BaseModel)
        into an instance of that model. If no input_model is available, the input is returned as provided,
        granted that it is a dict or a BaseModel instance.

        Args:
            inputs (Union[Dict[str, object], InputModelT]): The raw input data.

        Returns:
            Union[InputModelT, Dict[str, object]]: A validated model instance or the original input.

        Raises:
            ValueError: If a BaseModel input is not an instance of the expected input_model.
            TypeError: If the input is neither a dict nor a Pydantic model.
        """
        if self.input_model is not None:
            if isinstance(inputs, dict):
                return self.input_model.model_validate(inputs)
            if isinstance(inputs, BaseModel):
                if not isinstance(inputs, self.input_model):
                    raise ValueError(
                        f"Input model mismatch. Expected {self.input_model.__name__}, "
                        f"got {type(inputs).__name__}."
                    )
                return inputs
            raise TypeError(
                f"Inputs must be a dict or a Pydantic model, got {type(inputs).__name__}."
            )
        if isinstance(inputs, (dict, BaseModel)):
            return inputs
        raise TypeError(
            f"Inputs must be a dict or a Pydantic model, got {type(inputs).__name__}."
        )

    def validate_output(
        self, output: Union[Dict[str, object], OutputModelT]
    ) -> Union[OutputModelT, Dict[str, object]]:
        """Validates and parses the operator's raw output via the structured output model.

        If a structured_output model is defined, this method converts the raw output (either a dict or a BaseModel)
        into a validated instance of that model. Otherwise, it returns the raw output unmodified.

        Args:
            output (Union[Dict[str, object], OutputModelT]): The raw output data.

        Returns:
            Union[OutputModelT, Dict[str, object]]: A validated output model instance if structured_output is defined; otherwise, the raw output.

        Raises:
            ValueError: If a BaseModel output is not an instance of the expected structured_output type.
            TypeError: If output is neither a dict nor a Pydantic model when a structured_output model is defined.
        """
        if self.structured_output is not None:
            if isinstance(output, dict):
                return self.structured_output.model_validate(output)
            if isinstance(output, BaseModel):
                if not isinstance(output, self.structured_output):
                    raise ValueError(
                        f"Output model mismatch. Expected {self.structured_output.__name__}, "
                        f"got {type(output).__name__}."
                    )
                return output
            raise TypeError(
                f"Output must be a dict or a Pydantic model, got {type(output).__name__}."
            )
        return output
