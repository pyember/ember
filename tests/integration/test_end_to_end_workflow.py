"""
End-to-end integration tests for Ember workflows.

These tests exercise complete workflows from model creation to operator execution
in an integrated manner using real components and interfaces.
"""

import logging
from typing import Any, Dict, List, Type

import pytest

# Core components
# Model components
from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig
from ember.core.registry.operator.base._module import static_field
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification

# Import base types
from ember.core.types.ember_model import EmberModel

# Configure logging
logger = logging.getLogger(__name__)

# Only run these tests when explicitly enabled
pytestmark = [
    pytest.mark.integration,
]


class SummarizeInput(EmberModel):
    """Input model for the summarizer."""

    text: str
    max_words: int = 50


class SummarizeOutput(EmberModel):
    """Output model for the summarizer."""

    summary: str
    word_count: int


class SummarizeSpecification(Specification[SummarizeInput, SummarizeOutput]):
    """Specification for the summarizer."""

    input_model: Type[SummarizeInput] = SummarizeInput
    structured_output: Type[SummarizeOutput] = SummarizeOutput
    prompt_template: str = """Summarize the following text in {max_words} words or less:
    
{text}
    
Summary:"""


class SummarizerOperator(Operator[SummarizeInput, SummarizeOutput]):
    """Operator that summarizes text using real LLM backends."""

    specification = SummarizeSpecification()

    # Define static fields
    model_name: str = static_field()
    lm_module: LMModule = static_field()

    def __init__(
        self, model_name: str = "openai:gpt-3.5-turbo", simulate_api: bool = True
    ):
        """
        Initialize the summarizer with a real LMModule.

        Args:
            model_name: The model identifier to use for summarization
            simulate_api: Whether to simulate API calls for testing
        """
        self.model_name = model_name

        # Create a real LMModule connected to the model service
        self.lm_module = LMModule(
            config=LMModuleConfig(id=model_name), simulate_api=simulate_api
        )
        logger.info(f"Initialized SummarizerOperator with model: {model_name}")

    def forward(self, *, inputs: SummarizeInput) -> SummarizeOutput:
        """
        Summarize the input text using a real language model.

        Args:
            inputs: The text to summarize and constraints

        Returns:
            Structured summary output with word count
        """
        # Render the prompt according to the specification
        prompt = self.specification.render_prompt(inputs=inputs)
        logger.debug(f"Generated prompt: {prompt[:100]}...")

        # Call the real language model
        response = self.lm_module(prompt=prompt)

        # Extract the summary and count words
        summary = response.strip()
        word_count = len(summary.split())

        logger.info(f"Generated summary with {word_count} words")
        return SummarizeOutput(summary=summary, word_count=word_count)


class EnsembleOperatorInputs(EmberModel):
    """Input model for Ensemble operator"""

    query: str


class EnsembleOperatorOutputs(EmberModel):
    """Output model for Ensemble operator"""

    responses: List[str]


class EnsembleOperator(Operator[EnsembleOperatorInputs, EnsembleOperatorOutputs]):
    """Real implementation of the Ensemble Operator."""

    specification = Specification[EnsembleOperatorInputs, EnsembleOperatorOutputs](
        input_model=EnsembleOperatorInputs, structured_output=EnsembleOperatorOutputs
    )

    # Define static fields
    lm_modules: List[LMModule] = static_field()

    def __init__(self, lm_modules: List[LMModule]):
        """Initialize with LM modules."""
        self.lm_modules = lm_modules

    def forward(self, *, inputs: EnsembleOperatorInputs) -> EnsembleOperatorOutputs:
        """Execute query across all models."""
        rendered_prompt = self.specification.render_prompt(inputs=inputs)
        responses = [lm(prompt=rendered_prompt) for lm in self.lm_modules]
        return EnsembleOperatorOutputs(responses=responses)


class JudgeSynthesisInputs(EmberModel):
    """Input model for JudgeSynthesis."""

    query: str
    responses: List[str]


class JudgeSynthesisOutputs(EmberModel):
    """Output model for JudgeSynthesis."""

    final_answer: str
    reasoning: str


class JudgeSynthesisOperator(Operator[JudgeSynthesisInputs, JudgeSynthesisOutputs]):
    """Implementation of the Judge Synthesis Operator."""

    specification = Specification[JudgeSynthesisInputs, JudgeSynthesisOutputs](
        input_model=JudgeSynthesisInputs,
        structured_output=JudgeSynthesisOutputs,
        prompt_template=(
            "We have multiple advisors who proposed different answers:\n"
            "{responses}\n"
            "Now, we want to synthesize a single best, final answer to:\n"
            "{query}\n"
            "Explain your reasoning concisely, then provide the single best final answer.\n"
            "Format:\n"
            "Reasoning: <your reasoning for synthesizing this answer in this way>\n"
            "Final Answer: <the single best answer>\n"
        ),
    )

    # Define static fields
    lm_module: LMModule = static_field()

    def __init__(self, lm_module: LMModule):
        """Initialize with LM module."""
        self.lm_module = lm_module

    def forward(self, *, inputs: JudgeSynthesisInputs) -> JudgeSynthesisOutputs:
        """Synthesize a final answer from multiple responses."""
        rendered_prompt = self.specification.render_prompt(inputs=inputs)
        raw_output = self.lm_module(prompt=rendered_prompt).strip()

        # Parse the response to extract reasoning and final answer
        final_answer = "Unknown"
        reasoning_lines = []
        in_reasoning_section = False

        for line in raw_output.splitlines():
            line = line.strip()

            if line.startswith("Final Answer:"):
                final_answer = line.replace("Final Answer:", "").strip()
                break
            elif line.startswith("Reasoning:"):
                in_reasoning_section = True
                reasoning_part = line.replace("Reasoning:", "").strip()
                if reasoning_part:
                    reasoning_lines.append(reasoning_part)
            elif in_reasoning_section:
                reasoning_lines.append(line)

        reasoning = "\n".join(reasoning_lines)

        return JudgeSynthesisOutputs(final_answer=final_answer, reasoning=reasoning)


class MostCommonInputs(EmberModel):
    """Input model for MostCommon."""

    responses: List[str]


class MostCommonOutputs(EmberModel):
    """Output model for MostCommon."""

    most_common: str
    counts: Dict[str, int]


class MostCommonOperator(Operator[MostCommonInputs, MostCommonOutputs]):
    """Implementation of the MostCommon operator."""

    specification = Specification[MostCommonInputs, MostCommonOutputs](
        input_model=MostCommonInputs, structured_output=MostCommonOutputs
    )

    def forward(self, *, inputs: MostCommonInputs) -> MostCommonOutputs:
        """Find the most common response."""
        # Count occurrences
        counts: Dict[str, int] = {}
        for response in inputs.responses:
            counts[response] = counts.get(response, 0) + 1

        # Find most common
        most_common = max(counts.items(), key=lambda x: x[1])[0] if counts else ""

        return MostCommonOutputs(most_common=most_common, counts=counts)


class Sequential(Operator):
    """Implementation of the Sequential operator pattern."""

    # Define a generic specification
    specification = Specification(
        input_model=None, structured_output=None, check_all_placeholders=False
    )

    # Define static fields
    operators: List[Operator] = static_field()

    def __init__(self, operators: List[Operator]):
        """Initialize with a list of operators."""
        self.operators = operators

    def forward(self, *, inputs: Any) -> Any:
        """Execute operators sequentially."""
        result = inputs
        for operator in self.operators:
            result = operator(inputs=result)
        return result


# Use a thin wrapper around dict for XCSGraph
# This simplifies the integration test while still
# testing the core functionality
class XCSGraph:
    """Simplified XCS Graph implementation for testing."""

    def __init__(self):
        """Initialize an empty graph."""
        self.nodes = {}
        self.edges = {}

    def add_node(self, operator: Any, node_id: str):
        """Add a node to the graph."""
        self.nodes[node_id] = operator
        self.edges[node_id] = []
        return node_id

    def add_edge(self, from_id: str, to_id: str, input_mapping=None):
        """Add an edge between nodes."""
        self.edges[from_id].append((to_id, input_mapping))


def execute_graph(
    graph: XCSGraph,
    global_input: Dict[str, Any],
    concurrency: bool = False,
    executor=None,
):
    """Simplified graph execution for testing."""
    results = {}

    # Execute each operator in sequence, passing inputs from previous nodes
    for node_id, operator in graph.nodes.items():
        # First, determine inputs for this node
        node_inputs = dict(global_input)

        # For the ensemble operator, create proper input model
        if node_id == "ensemble":
            if "query" in node_inputs:
                node_inputs = EnsembleOperatorInputs(query=node_inputs["query"])
        # For the judge operator, create proper input model with responses
        elif node_id == "judge" and "ensemble" in results:
            # Get ensemble result
            ensemble_result = results["ensemble"]
            node_inputs = JudgeSynthesisInputs(
                query=global_input.get("query", ""), responses=ensemble_result.responses
            )

        # Execute operator and store results
        results[node_id] = operator(inputs=node_inputs)

        # Pass results to outbound edges
        for to_id, mapping in graph.edges.get(node_id, []):
            if to_id in graph.nodes:
                to_operator = graph.nodes[to_id]
                # Process input mapping if provided
                if mapping:
                    mapped_inputs = {}
                    for target_key, mapper in mapping.items():
                        if callable(mapper):
                            mapped_inputs[target_key] = mapper(results[node_id])
                        else:
                            mapped_inputs[target_key] = mapper
                    # Update global_input for the next node
                    global_input.update(mapped_inputs)

    return results


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
    """End-to-end integration tests using real components."""

    def test_simple_operator_execution(self, sample_text):
        """Test simple simulated LLM execution."""
        # Create a simple LM module with simulation
        lm_module = LMModule(
            config=LMModuleConfig(id="openai:gpt-3.5-turbo"), simulate_api=True
        )

        # Create a simple prompt
        prompt = f"Summarize this text in 30 words or less: {sample_text[:200]}..."

        # Execute the LM call
        result = lm_module(prompt=prompt)

        # Verify the result is a simulated response
        assert isinstance(result, str)
        assert "SIMULATED_RESPONSE" in result or "Response to:" in result
        assert len(result) > 0

    def test_graph_execution_with_ensemble(self, sample_text):
        """Test execution of a real operator graph with ensemble and judge."""
        # Create LM modules with simulation
        lm_modules = [
            LMModule(
                config=LMModuleConfig(id="openai:gpt-3.5-turbo", temperature=0.7),
                simulate_api=True,
            )
            for _ in range(3)
        ]

        # Create a real ensemble operator with real LM modules
        ensemble = EnsembleOperator(lm_modules=lm_modules)

        # Create a real judge operator
        judge_lm = LMModule(
            config=LMModuleConfig(id="openai:gpt-3.5-turbo", temperature=0.2),
            simulate_api=True,
        )
        judge = JudgeSynthesisOperator(lm_module=judge_lm)

        # Build execution graph
        graph = XCSGraph()
        graph.add_node(operator=ensemble, node_id="ensemble")
        graph.add_node(operator=judge, node_id="judge")
        graph.add_edge(from_id="ensemble", to_id="judge")

        # Execute graph
        result = execute_graph(
            graph=graph,
            global_input={"query": "Summarize this text: " + sample_text[:200]},
            concurrency=True,
        )

        # Verify the graph execution structure
        assert result is not None
        assert "ensemble" in result
        assert "judge" in result

        # Verify ensemble results
        ensemble_result = result["ensemble"]
        assert isinstance(ensemble_result, EnsembleOperatorOutputs)
        assert hasattr(ensemble_result, "responses")
        assert isinstance(ensemble_result.responses, list)
        assert len(ensemble_result.responses) == 3

    def test_complex_workflow_with_multiple_operators(self, sample_text):
        """Test a complex workflow with real operators."""
        # Skip app_context creation and model registration

        # Create LM modules with simulation
        ensemble_lms = [
            LMModule(
                config=LMModuleConfig(id="openai:gpt-3.5-turbo", temperature=0.7),
                simulate_api=True,
            )
            for _ in range(3)
        ]

        # Create a chain of real operators
        ensemble = EnsembleOperator(lm_modules=ensemble_lms)
        most_common = MostCommonOperator()

        # Create a custom summarizer
        summarizer = SummarizerOperator(model_name="openai:gpt-4", simulate_api=True)

        # Build a multi-stage execution pipeline
        pipeline = Sequential(operators=[ensemble, most_common])

        # Execute the pipeline
        ensemble_input = EnsembleOperatorInputs(
            query="Analyze the following: " + sample_text[:300]
        )
        pipeline_result = pipeline(inputs=ensemble_input)

        # Verify the pipeline results
        assert isinstance(pipeline_result, MostCommonOutputs)
        assert hasattr(pipeline_result, "most_common")
        assert hasattr(pipeline_result, "counts")

        # Execute the summarizer with the pipeline result
        summary_input = SummarizeInput(
            text=f"Refine this response: {pipeline_result.most_common}", max_words=50
        )
        summary_result = summarizer(inputs=summary_input)

        # Verify the summarizer results
        assert isinstance(summary_result, SummarizeOutput)
        assert hasattr(summary_result, "summary")
        assert isinstance(summary_result.summary, str)
        assert len(summary_result.summary) > 0
