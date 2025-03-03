from __future__ import annotations
from typing import Literal, Optional, Type

from pydantic import BaseModel

from ember.core.registry.prompt_specification.specification import Specification


class CaravanLabelsOutput(BaseModel):
    """
    Data model representing Caravan labeling output.

    Attributes:
        label (Literal["0", "1"]): The assigned label, where "0" indicates benign and "1" indicates malicious.
    """

    label: Literal["0", "1"]


class CaravanLabelingInputs(BaseModel):
    """
    Data model representing Caravan labeling inputs.

    Attributes:
        flow (str): The flow stream data used for labeling network traffic.
    """

    flow: str


class CaravanLabelingSpecification(Specification[CaravanLabelingInputs, CaravanLabelsOutput]):
    """
    Specification for labeling network flows as benign or malicious.

    This specification defines a prompt template for network security experts to label input flows.
    It leverages associated Pydantic models to validate both the inputs and the outputs.

    Attributes:
        prompt_template (str): The prompt template that incorporates input data via placeholders.
        structured_output (Optional[Type[CaravanLabelsOutput]]): The output model used for validating results.
        input_model (Optional[Type[CaravanLabelingInputs]]): The input model used for validating provided inputs.
    """

    prompt_template: str = (
        "You are a network security expert.\n"
        "Given these unlabeled flows:\n{flow}\n"
        "Label each flow as 0 for benign or 1 for malicious, one per line, no explanation.\n"
    )
    structured_output: Optional[Type[CaravanLabelsOutput]] = CaravanLabelsOutput
    input_model: Optional[Type[CaravanLabelingInputs]] = CaravanLabelingInputs