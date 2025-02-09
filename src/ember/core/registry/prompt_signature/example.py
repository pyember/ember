from __future__ import annotations
from typing import Literal, Optional, Type

from pydantic import BaseModel

from ember.core.registry.prompt_signature.signatures import Signature


class CaravanLabelsOutput(BaseModel):
    """Pydantic model for Caravan labeling output.

    Attributes:
        label (Literal["0", "1"]): The label assigned to the input. It must be "0" (benign) or "1" (malicious).
    """

    label: Literal["0", "1"]


class CaravanLabelingInputs(BaseModel):
    """Pydantic model for Caravan labeling inputs.

    Attributes:
        question (str): The question text used for labeling.
    """

    question: str


class CaravanLabelingSignature(Signature):
    """Signature for labeling network flows as benign or malicious.

    Attributes:
        prompt_template (str): The prompt template for processing the input.
        structured_output (Optional[Type[BaseModel]]): Output model class used to validate results.
        input_model (Optional[Type[BaseModel]]): Input model class used to validate inputs.
    """

    prompt_template: str = (
        "You are a network security expert.\n"
        "Given these unlabeled flows:\n{question}\n"
        "Label each flow as 0 for benign or 1 for malicious, one per line, no explanation.\n"
    )
    structured_output: Optional[Type[BaseModel]] = CaravanLabelsOutput
    input_model: Optional[Type[BaseModel]] = CaravanLabelingInputs


# Example usage with an ensemble operator:
# from ember.registry.operators.operator_registry import EnsembleOperator, EnsembleOperatorInputs
# from ember.registry.operators.operator_base import LMModuleConfig, LMModule
#
# question_data = "What is the capital of France?"
#
# # Define LMModule configurations
# lm_configs = [
#     LMModuleConfig(model_name="gpt-4o", temperature=0.7) for _ in range(3)
# ]
#
# # Create LMModules from the configurations
# lm_modules = [LMModule(config=config) for config in lm_configs]
#
# # Create an instance of the EnsembleOperator with the LMModules and signature
# caravan_operator = EnsembleOperator(lm_modules=lm_modules)
#
# # Build the inputs using the signature's input model
# inputs = caravan_operator.build_inputs(query=question_data)
#
# # Execute the operator
# result = caravan_operator(inputs=inputs)
# print(result)
