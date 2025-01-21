from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, model_validator


class Signature(BaseModel):
    """
    Base class for operator signatures.

    Attributes:
        required_inputs: A list of input field names required by the operator.
        prompt_template: A template string that may use the required inputs, if applicable.
        structured_output: An optional Pydantic model class used to validate and structure the output.
        input_model: An optional Pydantic model class defining the required input fields.
    """

    required_inputs: List[str]
    prompt_template: Optional[str] = None
    structured_output: Optional[Type[BaseModel]] = None
    input_model: Optional[Type[BaseModel]] = None

    @model_validator(mode="after")
    def check_template(cls, values: Any) -> Any:
        template = values.prompt_template
        required = values.required_inputs
        if template:
            for inp in required:
                if f"{{{inp}}}" not in template:
                    raise ValueError(
                        f"Required input '{inp}' not found in prompt_template."
                    )
        return values

    def model_json_schema(self) -> dict:
        """
        Return JSON schema of the input model if defined. Useful for introspection and documentation.
        """
        if self.input_model:
            return self.input_model.model_json_schema()
        return {}

    def validate_inputs(self, inputs: Union[Dict[str, Any], BaseModel]) -> BaseModel:
        """
        The canonical place to parse or coerce raw inputs into
        a validated Pydantic model, using self.input_model if present.

        If no input_model is defined, this method returns the input as-is (no validation),
        provided it's already a dict or a Pydantic model. Otherwise, a TypeError is raised.
        """
        if self.input_model:
            if isinstance(inputs, dict):
                return self.input_model(**inputs)
            elif isinstance(inputs, BaseModel):
                if not isinstance(inputs, self.input_model):
                    raise ValueError(
                        f"Input model mismatch. Expected {self.input_model.__name__}, got {type(inputs).__name__}"
                    )
                return inputs
            else:
                raise TypeError(
                    f"Inputs must be a dict or a Pydantic model, got {type(inputs).__name__}"
                )
        else:
            # Fallback if no input_model is defined:
            if isinstance(inputs, dict):
                return inputs
            elif isinstance(inputs, BaseModel):
                return inputs
            else:
                raise TypeError(
                    f"Inputs must be a dict or a Pydantic model, got {type(inputs).__name__}"
                )

    def validate_output(self, output: Any) -> Any:
        """
        The canonical place to parse or coerce the operator's raw
        output into a structured Pydantic model, using self.structured_output if present.

        If no structured_output is defined, the raw output is returned as-is.
        """
        if self.structured_output:
            if isinstance(output, dict):
                return self.structured_output(**output)
            elif isinstance(output, BaseModel):
                if not isinstance(output, self.structured_output):
                    raise ValueError(
                        f"Output model mismatch. Expected {self.structured_output.__name__}, got {type(output).__name__}"
                    )
                return output
            else:
                raise TypeError(
                    f"Output must be a dict or a Pydantic model, got {type(output).__name__}"
                )
        else:
            return output


class CaravanLabelsOutput(BaseModel):
    """Example output model for Caravan labeling."""

    label: str


class CaravanLabelingInputs(BaseModel):
    question: str


class CaravanLabelingSignature(Signature):
    """Signature for labeling network flows as benign or malicious."""

    required_inputs: List[str] = ["question"]
    prompt_template: str = (
        "You are a network security expert.\n"
        "Given these unlabeled flows:\n{question}\n"
        "Label each flow as 0 for benign or 1 for malicious, one per line, no explanation.\n"
    )
    structured_output: Optional[Type[BaseModel]] = CaravanLabelsOutput
    input_model: Optional[Type[BaseModel]] = CaravanLabelingInputs


# Example usage with an ensemble operator:
# from avior.registry.operators.operator_registry import EnsembleOperator, EnsembleOperatorInputs
# from avior.registry.operators.operator_base import LMModuleConfig, LMModule

# question_data = "What is the capital of France?"

# # Define LMModule configurations
# lm_configs = [
#     LMModuleConfig(model_name="gpt-4o", temperature=0.7) for _ in range(3)
# ]

# # Create LMModules from the configurations
# lm_modules = [LMModule(config=config) for config in lm_configs]

# # Create an instance of the EnsembleOperator with the LMModules and signature
# caravan_operator = EnsembleOperator(
#     lm_modules=lm_modules,
# )

# # Build the inputs using the signature's input model
# inputs = caravan_operator.build_inputs(query=question_data)

# # Execute the operator
# result = caravan_operator(inputs=inputs)
# print(result)
