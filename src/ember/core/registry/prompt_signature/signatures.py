from __future__ import annotations
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel, model_validator


class Signature(BaseModel):
    """Base class representing an operator's signature.

    Attributes:
        prompt_template (Optional[str]): Template string that may reference input names.
        structured_output (Optional[Type[BaseModel]]): Pydantic model class for validating the output.
        input_model (Optional[Type[BaseModel]]): Pydantic model class defining the expected input fields.
        check_all_placeholders (bool): Flag to enforce the inclusion of all required placeholders.
    """

    prompt_template: Optional[str] = None
    structured_output: Optional[Type[BaseModel]] = None
    input_model: Optional[Type[BaseModel]] = None
    check_all_placeholders: bool = False

    @model_validator(mode="after")
    @classmethod
    def check_template(cls: Type[Signature], values: Dict[str, Any]) -> Dict[str, Any]:
        """Validates that the prompt_template contains placeholders for every required input field.

        This validator inspects the template provided via the 'prompt_template' field and ensures it
        references every required field specified by 'input_model'. If any required placeholder is missing,
        a ValueError is raised.

        Args:
            values (Dict[str, Any]): Dictionary mapping field names to their corresponding values.

        Returns:
            Dict[str, Any]: The validated dictionary of field values.

        Raises:
            ValueError: If a required placeholder for an input field is absent in prompt_template.
        """
        template: Optional[str] = values.get("prompt_template")
        input_model: Optional[Type[BaseModel]] = values.get("input_model")
        check_all: bool = values.get("check_all_placeholders", False)
        if template is not None and input_model is not None and check_all:
            for field_name, field in input_model.model_fields.items():
                if field.required:
                    placeholder: str = f"{{{field_name}}}"
                    if placeholder not in template:
                        raise ValueError(
                            f"Required input '{field_name}' not found in prompt_template."
                        )
        return values

    def render_prompt(self, inputs: Dict[str, Any]) -> str:
        """Renders a prompt string based on provided inputs and configuration.

        If a prompt_template is defined, the method formats it using the values from 'inputs'.
        Otherwise, if an input_model is available, it concatenates the values of the required fields.
        If neither is available, a ValueError is raised.

        Args:
            inputs (Dict[str, Any]): Dictionary containing input values.

        Returns:
            str: The prompt string generated from the inputs.

        Raises:
            ValueError: If neither a prompt_template nor an input_model is defined.
        """
        if self.prompt_template:
            return self.prompt_template.format(**inputs)
        if self.input_model:
            required_fields = [
                name
                for name, field in self.input_model.model_fields.items()
                if field.required
            ]
            return "\n".join(str(inputs.get(field, "")) for field in required_fields)
        raise ValueError(
            "No prompt_template or input_model defined for rendering prompt."
        )

    def model_json_schema(self) -> Dict[str, Any]:
        """Retrieves the JSON schema for the input model.

        If an input_model is defined, its JSON schema is returned; otherwise, an empty dictionary is provided.

        Returns:
            Dict[str, Any]: The JSON schema of the input_model or an empty dict if not defined.
        """
        if self.input_model is not None:
            return self.input_model.model_json_schema()
        return {}

    def validate_inputs(
        self, inputs: Union[Dict[str, Any], BaseModel]
    ) -> Union[BaseModel, Dict[str, Any]]:
        """Validates and parses raw inputs using the defined input model.

        When an input_model is specified, this method converts the raw input (either a dict or BaseModel)
        into an instance of that model. If no input_model is available, the input is returned as provided,
        granted that it is a dict or a BaseModel instance.

        Args:
            inputs (Union[Dict[str, Any], BaseModel]): The raw input data.

        Returns:
            Union[BaseModel, Dict[str, Any]]: A validated model instance or the original input.

        Raises:
            ValueError: If a BaseModel input is not an instance of the expected input_model.
            TypeError: If the input is neither a dict nor a BaseModel.
        """
        if self.input_model is not None:
            if isinstance(inputs, dict):
                return self.input_model.model_validate(obj=inputs)
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

    def validate_output(self, output: Any) -> Any:
        """Validates and parses the operator's raw output via the structured output model.

        If a structured_output model is defined, this method converts the raw output (either a dict or a BaseModel)
        into a validated instance of that model. Otherwise, it returns the raw output unmodified.

        Args:
            output (Any): The raw output data.

        Returns:
            Any: A validated output model instance if structured_output is defined; otherwise, the raw output.

        Raises:
            ValueError: If a BaseModel output is not an instance of the expected structured_output type.
            TypeError: If output is neither a dict nor a BaseModel when a structured_output model is defined.
        """
        if self.structured_output is not None:
            if isinstance(output, dict):
                return self.structured_output.model_validate(obj=output)
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
