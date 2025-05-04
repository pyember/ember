"""
Prompt Enhancement Pattern Implementation

This module implements the PromptEnhancer pattern, a powerful mechanism for
transforming sparse user queries into comprehensive, detailed prompts that
capture the likely user intent.
"""

from __future__ import annotations
from typing import Type

from ember.core.registry.model.model_module.lm import LMModule
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel


class PromptEnhancerOperatorInputs(EmberModel):
    """
    Structured input model for prompt enhancement operations.

    Attributes:
        query: The original user query that needs enhancement.
    """
    query: str


class PromptEnhancerOperatorOutputs(EmberModel):
    """
    Structured output model for prompt enhancement results.

    Attributes:
        enhanced_query: The enhanced, more detailed query.
    """
    enhanced_query: str


class PromptEnhancerSpecification(Specification):
    """Specification for PromptEnhancer defining the enhancement template."""
    prompt_template: str = (
        "You are a query enhancement expert. Your task is to expand the following brief query into a more detailed, "
        "comprehensive query that captures the likely intent of the user.\n\n"
        "Original query: {query}\n\n"
        "Provide an enhanced version that includes relevant context, specifications, and considerations. "
        "Focus on expanding the query without changing its core meaning or intent.\n\n"
        "Enhanced query:"
    )
    input_model: Type[EmberModel] = PromptEnhancerOperatorInputs
    structured_output: Type[EmberModel] = PromptEnhancerOperatorOutputs


class PromptEnhancerOperator(Operator[PromptEnhancerOperatorInputs, PromptEnhancerOperatorOutputs]):
    """
    Enhances a sparse query into a comprehensive, detailed prompt.

    This operator transforms short user queries into detailed prompts that 
    better capture the user's intent, leading to more accurate and relevant 
    responses from language models.
    """

    specification: Specification = PromptEnhancerSpecification()
    lm_module: LMModule

    def __init__(self, *, lm_module: LMModule) -> None:
        """
        Initializes the prompt enhancer with a language model module.

        Args:
            lm_module: Language model module to execute the enhancement operation.
                      This module must conform to the LMModule interface.
        """
        self.lm_module = lm_module

    def forward(self, *, inputs: PromptEnhancerOperatorInputs) -> PromptEnhancerOperatorOutputs:
        """
        Enhances the input query into a more detailed prompt.

        Args:
            inputs: Contains the original user query.

        Returns:
            Enhanced query that better captures the user's intent.
        """
        enhanced_query = self.lm_module(
            prompt=self.specification.render_prompt(inputs={"query": inputs.query})
        )
        return PromptEnhancerOperatorOutputs(enhanced_query=enhanced_query)