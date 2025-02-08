from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, model_validator


class Signature(BaseModel):
    """Base class for operator signatures.

    Attributes:
        required_inputs (List[str]): List of field names required by the operator.
        prompt_template (Optional[str]): Template string, potentially referencing required input names.
        structured_output (Optional[Type[BaseModel]]): Pydantic model class used to validate and structure output.
        input_model (Optional[Type[BaseModel]]): Pydantic model class defining the required input fields.
    """

    required_inputs: List[str]
    prompt_template: Optional[str] = None
    structured_output: Optional[Type[BaseModel]] = None
    input_model: Optional[Type[BaseModel]] = None

    @model_validator(mode="after")
    @classmethod
    def check_template(cls: Type[Signature], values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the prompt_template includes all required input placeholders.

        Args:
            values (Dict[str, Any]): Dictionary of model field values.

        Returns:
            Dict[str, Any]: The validated field values.

        Raises:
            ValueError: If a required input placeholder is missing in the prompt_template.
        """
        template: Optional[str] = values.get("prompt_template")
        required_inputs: List[str] = values.get("required_inputs", [])
        if template is not None:
            for input_name in required_inputs:
                placeholder: str = f"{{{input_name}}}"
                if placeholder not in template:
                    raise ValueError(
                        f"Required input '{input_name}' not found in prompt_template."
                    )
        return values

    def model_json_schema(self) -> Dict[str, Any]:
        """Retrieve the JSON schema for the input model.

        Returns:
            Dict[str, Any]: The JSON schema of the input_model if defined, otherwise an empty dict.
        """
        if self.input_model is not None:
            return self.input_model.model_json_schema()
        return {}

    def validate_inputs(
        self, inputs: Union[Dict[str, Any], BaseModel]
    ) -> Union[BaseModel, Dict[str, Any]]:
        """Parse and validate raw inputs using the defined input_model.

        If an input_model is specified, this method converts raw inputs to an instance of that model.
        If no input_model is defined, the input is returned as-is provided it is either a dict or a BaseModel.

        Args:
            inputs (Union[Dict[str, Any], BaseModel]): Raw input data.

        Returns:
            Union[BaseModel, Dict[str, Any]]: A validated input model instance or the original input.

        Raises:
            ValueError: If the provided BaseModel is not an instance of the expected input_model.
            TypeError: If inputs is neither a dict nor a BaseModel.
        """
        if self.input_model is not None:
            if isinstance(inputs, dict):
                return self.input_model.model_validate(obj=inputs)
            elif isinstance(inputs, BaseModel):
                if not isinstance(inputs, self.input_model):
                    raise ValueError(
                        f"Input model mismatch. Expected {self.input_model.__name__}, "
                        f"got {type(inputs).__name__}."
                    )
                return inputs
            else:
                raise TypeError(
                    f"Inputs must be a dict or a Pydantic model, got {type(inputs).__name__}."
                )
        if isinstance(inputs, (dict, BaseModel)):
            return inputs
        raise TypeError(
            f"Inputs must be a dict or a Pydantic model, got {type(inputs).__name__}."
        )

    def validate_output(self, output: Any) -> Any:
        """Parse and validate the operator's raw output via the structured_output model.

        If structured_output is defined, converts raw output into a validated Pydantic model.
        Otherwise, returns the raw output unmodified.

        Args:
            output (Any): Raw output data.

        Returns:
            Any: A validated output model instance if structured_output is defined, or the raw output.

        Raises:
            ValueError: If the provided BaseModel does not match the expected structured_output.
            TypeError: If output is neither a dict nor a BaseModel when structured_output is defined.
        """
        if self.structured_output is not None:
            if isinstance(output, dict):
                return self.structured_output.model_validate(obj=output)
            elif isinstance(output, BaseModel):
                if not isinstance(output, self.structured_output):
                    raise ValueError(
                        f"Output model mismatch. Expected {self.structured_output.__name__}, "
                        f"got {type(output).__name__}."
                    )
                return output
            else:
                raise TypeError(
                    f"Output must be a dict or a Pydantic model, got {type(output).__name__}."
                )
        return output