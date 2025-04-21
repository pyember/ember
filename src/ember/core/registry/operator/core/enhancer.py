from typing import Type
from ember.core.exceptions import MissingLMModuleError
from ember.core.registry.model.model_module.lm import LMModule
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel


class PromptEnhancerOperatorInputs(EmberModel):
    """
    Input model for PromptEnhancerOperator.

    Attributes:
        query (str): The query string to enhance.
    """

    query: str


class PromptEnhancerOperatorOutputs(EmberModel):
    """
    Output model for PromptEnhancerOperator.

    Attributes:
        query (str): The enhanced query string.
    """

    query: str


class PromptEnhancerSpecification(Specification):
    """Specification for PromptEnhancer defining the query enhancement prompt."""

    prompt_template: str = (
        "You are a elaborator of prompt to enhance meaning given sparse input.\n"
        "Given this initial prompt: {query}\n"
        "Your objective is to reason through the provided query to intuit potential query expansion.\n"
        "Please provide ample reasoning to the question 'If I were providing this query, what would I want to know'.\n"
        "Use slow and thoughtful reasoning chains. Provide:\n"
        "Reasoning: <Your reasoning>\n"
        "Enhanced Query: <New, thorough, and improved query based on previous inputted query>"
    )
    input_model: Type[EmberModel] = PromptEnhancerOperatorInputs
    structured_output: Type[EmberModel] = PromptEnhancerOperatorOutputs


class PromptEnhancerOperator(
    Operator[PromptEnhancerOperatorInputs, PromptEnhancerOperatorOutputs]
):
    """Operator to enhance a given query and provide an improved query."""

    specification: Specification = PromptEnhancerSpecification()
    lm_module: LMModule

    def __init__(self, *, lm_module: LMModule) -> None:
        self.lm_module = lm_module

    def forward(
        self, *, inputs: PromptEnhancerOperatorInputs
    ) -> PromptEnhancerOperatorOutputs:
        if not self.lm_module:
            raise MissingLMModuleError(
                "No LM module attached to PromptEnhancerOperator."
            )
        rendered_prompt: str = self.specification.render_prompt(inputs=inputs)
        raw_output: str = self.lm_module(prompt=rendered_prompt).strip()

        # Initialize default values
        enhanced_query = ""

        # Process each line with robust parsing
        in_reasoning_section = False
        in_enhanced_query_section = False
        enhanced_query_lines = []

        for line in raw_output.split("\n"):
            clean_line = line.strip()

            # Parse reasoning section (we don't need to capture this)
            if clean_line.startswith("Reasoning:"):
                in_reasoning_section = True
                in_enhanced_query_section = False

            # Parse enhanced query
            elif clean_line.startswith("Enhanced Query:"):
                in_reasoning_section = False
                in_enhanced_query_section = True
                enhanced_part = clean_line.replace("Enhanced Query:", "").strip()
                if enhanced_part:
                    enhanced_query_lines.append(enhanced_part)

            # Continue parsing multi-line sections
            elif in_enhanced_query_section:
                enhanced_query_lines.append(clean_line)

        # Finalize parsing
        if enhanced_query_lines:
            enhanced_query = "\n".join(enhanced_query_lines)
        else:
            # Fallback if no enhanced query was found
            enhanced_query = inputs.query

        # Return as dictionary - the operator __call__ will properly convert to model
        return {
            "query": enhanced_query,
        }
