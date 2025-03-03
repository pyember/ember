"""
End-to-end integration tests for Ember workflows.

These tests exercise complete workflows from model creation to operator execution
in an integrated manner.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, ANY

from ember.core.app_context import create_ember_app
from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig
from ember.core.registry.operator.base import Operator
from ember.core.registry.prompt_specification.specification import Specification
from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.engine import execute_graph
from ember.core.non import UniformEnsemble
from ember.core.non import JudgeSynthesis

from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Type


# Only run these tests when explicitly enabled
pytestmark = [
    pytest.mark.integration,
]


class SummarizeInput(BaseModel):
    """Input model for the summarizer."""
    text: str
    max_words: int = 50


class SummarizeOutput(BaseModel):
    """Output model for the summarizer."""
    summary: str
    word_count: int


class SummarizeSpecification(Specification):
    """Specification for the summarizer."""
    input_model: Type[SummarizeInput] = SummarizeInput
    structured_output: Type[SummarizeOutput] = SummarizeOutput
    prompt_template: str = """Summarize the following text in {max_words} words or less:
    
{text}
    
Summary:"""


class SummarizerOperator(Operator[SummarizeInput, SummarizeOutput]):
    """Operator that summarizes text."""
    specification = SummarizeSpecification()
    
    def __init__(self, model_name: str = "mock:model"):
        """Initialize the summarizer."""
        self.model_name = model_name
        self.lm_module = LMModule(config=LMModuleConfig(id=model_name))
    
    def forward(self, *, inputs: SummarizeInput) -> SummarizeOutput:
        """Summarize the input text."""
        response = self.lm_module(self.specification.render_prompt(inputs=inputs))
        
        # Extract the summary and count words
        summary = response.strip()
        word_count = len(summary.split())
        
        return SummarizeOutput(summary=summary, word_count=word_count)


@pytest.fixture
def mock_model_response():
    """Mock model response for testing."""
    return "This is a generated summary of the text. It is concise and accurate."


@pytest.fixture
def sample_text():
    """Sample text for summarization testing."""
    return """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
    incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud 
    exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute 
    irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla 
    pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia 
    deserunt mollit anim id est laborum.
    
    Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque 
    laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi 
    architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas 
    sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione 
    voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet.
    """


class TestEndToEndWorkflows:
    """End-to-end integration tests for Ember workflows."""
    
    @pytest.mark.skipif(
        not os.environ.get("ALLOW_EXTERNAL_API_CALLS"),
        reason="External API calls not enabled"
    )
    def test_simple_operator_execution(self, sample_text):
        """Test the execution of a simple operator with real model calls."""
        # This test requires external API calls
        # Create app context
        app_context = create_ember_app()
        
        # Create the operator
        summarizer = SummarizerOperator(model_name="openai:gpt-3.5-turbo")
        
        # Execute the operator
        result = summarizer(inputs=SummarizeInput(text=sample_text, max_words=30))
        
        # Verify the result
        assert isinstance(result, SummarizeOutput)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0
        assert result.word_count <= 30  # Should respect max_words
    
    @patch("ember.core.registry.model.model_module.lm.LMModule.__call__")
    def test_graph_execution_with_ensemble(self, mock_lm_call, sample_text, mock_model_response):
        """Test the execution of a graph with ensemble and judge operators."""
        # Configure mock
        mock_lm_call.return_value = mock_model_response
        
        # Create operators
        ensemble = UniformEnsemble(num_units=3, model_name="mock:model", temperature=0.7)
        judge = JudgeSynthesis(model_name="mock:model")
        
        # Build execution graph
        graph = XCSGraph()
        graph.add_node(operator=ensemble, node_id="ensemble")
        graph.add_node(operator=judge, node_id="judge")
        graph.add_edge(from_id="ensemble", to_id="judge")
        
        # Execute graph
        from ember.xcs.engine.execution_options import execution_options
        
        with execution_options(max_workers=3):
            result = execute_graph(
                graph=graph,
                global_input={"query": sample_text},
                concurrency=True
            )
        
        # Verify the graph execution
        assert result is not None
        # The result object has changed structure, it's now a dictionary with node_ids as keys
        assert "ensemble" in result
        assert "judge" in result
        assert mock_lm_call.call_count >= 4  # 3 for ensemble + 1 for judge
    
    @patch("ember.core.registry.model.model_module.lm.LMModule.__call__")
    def test_complex_workflow_with_custom_operators(self, mock_lm_call, sample_text, mock_model_response):
        """Test a more complex workflow with custom operators and multiple stages."""
        # Configure mock
        mock_lm_call.return_value = mock_model_response
        
        # Create operators
        # In this test, we'll just use an ensemble and judge, like in the previous test
        # The test demonstrates how to use the execution options context manager
        ensemble = UniformEnsemble(num_units=3, model_name="mock:model", temperature=0.7)
        judge = JudgeSynthesis(model_name="mock:judge")
        
        # Build execution graph
        graph = XCSGraph()
        graph.add_node(operator=ensemble, node_id="ensemble")
        graph.add_node(operator=judge, node_id="judge")
        graph.add_edge(from_id="ensemble", to_id="judge")
        
        # Execute graph
        from ember.xcs.engine.execution_options import execution_options
        
        with execution_options(max_workers=3):
            result = execute_graph(
                graph=graph,
                global_input={
                    "query": sample_text
                },
                concurrency=True
            )
        
        # Verify the graph execution
        assert result is not None
        # The result object has changed structure, it's now a dictionary with node_ids as keys
        assert "ensemble" in result
        assert "judge" in result
        assert mock_lm_call.call_count >= 4  # 3 for ensemble + 1 judge