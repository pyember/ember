"""
Real integration tests for Ember workflows.

These tests use real Operator and ModelService implementations
to test the integration of components within the framework.
"""

from typing import Any, Callable, Dict, List, Optional

# Use the real model components
from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig

# Use the real operator and specifications
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification

# Core components
from ember.core.types.ember_model import EmberModel


class SummarizeInput(EmberModel):
    """Input model for summarization operator."""

    text: str
    max_words: int = 50


class SummarizeOutput(EmberModel):
    """Output model for summarization operator."""

    summary: str
    word_count: int


from ember.core.registry.operator.base._module import ember_field, static_field


class SummarizerOperator(Operator[SummarizeInput, SummarizeOutput]):
    """Simple operator that summarizes text using an LLM.

    Implements immutable dataclass fields with static/dynamic field designations
    and specification-driven execution.
    """

    # Define the class variable specification
    specification = Specification[SummarizeInput, SummarizeOutput](
        prompt_template="""Summarize the following text in {max_words} words or less:
        
{text}
        
Summary:""",
        input_model=SummarizeInput,
        structured_output=SummarizeOutput,
    )

    # Define static operator fields
    lm_module: LMModule = static_field()
    summarizer_config: dict = static_field(default_factory=dict)

    # Use ember_field for fields with converters or special initialization
    stats: dict = ember_field(
        static=True,  # Static field
        init=False,  # Not included in __init__ parameters
        default_factory=dict,  # Default value if not explicitly initialized
    )

    def __init__(
        self, *, model_id: str = "openai:gpt-3.5-turbo", simulate_api: bool = True
    ):
        """Initialize with a language model.

        Sets up the immutable fields. All fields are frozen after initialization.
        """
        self.lm_module = LMModule(
            config=LMModuleConfig(id=model_id), simulate_api=simulate_api
        )

        self.summarizer_config = {
            "model_id": model_id,
            "simulate_api": simulate_api,
            "max_output_length": 2000,
        }

        self._init_field(
            field_name="stats",
            value={
                "calls": 0,
                "total_tokens": 0,
                "created_at": "2025-03-11",
            },
        )

    def forward(self, *, inputs: SummarizeInput) -> SummarizeOutput:
        """Summarize the text using the language model.

        Implements the core computation logic as a pure functional transform
        from inputs to outputs without mutating instance state.
        """
        prompt = self.specification.render_prompt(inputs=inputs)

        max_length = self.summarizer_config.get("max_output_length", 2000)
        if len(inputs.text) > max_length:
            truncated_text = inputs.text[:max_length] + "..."
            inputs = SummarizeInput(text=truncated_text, max_words=inputs.max_words)
            prompt = self.specification.render_prompt(inputs=inputs)

        response = self.lm_module(prompt=prompt)

        summary = response.strip()
        word_count = len(summary.split())

        return SummarizeOutput(summary=summary, word_count=word_count)


class TranslateInput(EmberModel):
    """Input model for translation operator."""

    text: str
    target_language: str


class TranslateOutput(EmberModel):
    """Output model for translation operator."""

    translated_text: str
    detected_language: Optional[str] = None


class TranslatorOperator(Operator[TranslateInput, TranslateOutput]):
    """Simple operator that translates text.

    Implements operator pattern with:
    - Static fields for configuration
    - Pure functional forward method
    """

    # Define the specification as a class variable
    specification = Specification[TranslateInput, TranslateOutput](
        prompt_template="""Translate the following text to {target_language}:
        
{text}
        
Translation:""",
        input_model=TranslateInput,
        structured_output=TranslateOutput,
    )

    # Define static fields for resources and configuration
    lm_module: LMModule = static_field()  # Resource field
    model_config: dict = static_field(default_factory=dict)  # Configuration field

    def __init__(
        self, *, model_id: str = "openai:gpt-3.5-turbo", simulate_api: bool = True
    ):
        """Initialize with a language model.

        Sets up immutable operator fields.
        """
        self.lm_module = LMModule(
            config=LMModuleConfig(id=model_id), simulate_api=simulate_api
        )

        self.model_config = {
            "model_id": model_id,
            "simulate_api": simulate_api,
            "supports_language_detection": True,
        }

    def forward(self, *, inputs: TranslateInput) -> TranslateOutput:
        """Translate the text using the language model.

        Pure functional transform from inputs to outputs.
        """
        prompt = self.specification.render_prompt(inputs=inputs)

        response = self.lm_module(prompt=prompt)

        detected_language = None
        if self.model_config["supports_language_detection"] and len(inputs.text) > 20:
            detected_language = "English"  # Simulated detection

        return TranslateOutput(
            translated_text=response.strip(), detected_language=detected_language
        )


class SequentialPipeline:
    """A simple sequential pipeline of operators with input mappings."""

    def __init__(
        self,
        operators: List[Operator],
        input_mappings: Optional[List[Dict[str, Callable]]] = None,
    ):
        """
        Initialize with a list of operators and optional input mappings.

        Args:
            operators: List of operators to execute in sequence
            input_mappings: List of mapping functions to transform output of one operator
                          to input of the next. Should be one less than the number of operators.
        """
        self.operators = operators
        self.input_mappings = input_mappings or []

        # Ensure we have the right number of mappings
        if self.input_mappings and len(self.input_mappings) != len(operators) - 1:
            raise ValueError(
                f"Expected {len(operators) - 1} input mappings, got {len(self.input_mappings)}"
            )

    def run(self, inputs: Any) -> Any:
        """
        Run the full pipeline with input mappings.

        Args:
            inputs: Initial input for the first operator

        Returns:
            Output from the last operator
        """
        result = inputs

        for i, operator in enumerate(self.operators):
            if i > 0 and i - 1 < len(self.input_mappings):
                mapping = self.input_mappings[i - 1]
                mapped_input = {}
                for key, mapper in mapping.items():
                    mapped_input[key] = mapper(result)
                result = mapped_input

            result = operator(inputs=result)

        return result


class TestRealIntegration:
    """Integration tests using real components."""

    def test_lm_module_simulated_call(self):
        """Test that we can create and call an LM module with simulated API."""
        lm_module = LMModule(
            config=LMModuleConfig(id="openai:gpt-3.5-turbo"), simulate_api=True
        )

        result = lm_module(prompt="What is the capital of France?")

        assert isinstance(result, str)
        assert (
            "SIMULATED_RESPONSE" in result
        ), f"Expected simulated response, got: {result}"

    def test_parallel_lm_calls(self):
        """Test multiple LM modules can be created and called in parallel."""
        modules = [
            LMModule(
                config=LMModuleConfig(
                    id="openai:gpt-3.5-turbo", temperature=0.7 + i * 0.1
                ),
                simulate_api=True,
            )
            for i in range(3)
        ]

        prompt = "Tell me a joke about programming."
        responses = [module(prompt=prompt) for module in modules]

        assert len(responses) == 3

        for i, response in enumerate(responses):
            assert isinstance(response, str)
            assert (
                "SIMULATED_RESPONSE" in response
            ), f"Module {i} did not return simulated response"

    def test_config_variations(self):
        """Test LM modules with different configurations."""
        configs = [
            LMModuleConfig(id="openai:gpt-3.5-turbo"),
            LMModuleConfig(id="anthropic:claude-3-haiku", temperature=0.5),
            LMModuleConfig(
                id="anthropic:claude-3-sonnet",
                temperature=0.2,
                max_tokens=100,
                cot_prompt="Think step by step.",
            ),
        ]

        modules = [LMModule(config=config, simulate_api=True) for config in configs]

        for i, module in enumerate(modules):
            result = module(prompt=f"Test prompt {i}")
            assert isinstance(result, str)
            assert "SIMULATED_RESPONSE" in result

    def test_chat_simulation(self):
        """Test a simulated chat conversation."""
        module = LMModule(
            config=LMModuleConfig(
                id="openai:gpt-4o", persona="You are a helpful assistant."
            ),
            simulate_api=True,
        )

        messages = [
            "Hello, how are you?",
            "Can you help me with Python?",
            "How do I sort a list in reverse order?",
        ]

        responses = []
        for message in messages:
            response = module(prompt=message)
            responses.append(response)

        assert len(responses) == len(messages)
        for response in responses:
            assert isinstance(response, str)
            assert "SIMULATED_RESPONSE" in response

    def test_summarizer_operator(self):
        """Test a real summarizer operator with simulated API."""
        sample_text = """
        The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet.
        Amazingly few discotheques provide jukeboxes. How vexingly quick daft zebras jump!
        Pack my box with five dozen liquor jugs. The five boxing wizards jump quickly.
        """

        summarizer = SummarizerOperator(simulate_api=True)

        input_data = SummarizeInput(text=sample_text, max_words=10)

        result = summarizer(inputs=input_data)

        assert isinstance(result, SummarizeOutput)
        assert hasattr(result, "summary")
        assert isinstance(result.summary, str)
        assert hasattr(result, "word_count")
        assert isinstance(result.word_count, int)

    def test_translator_operator(self):
        """Test a real translator operator with simulated API."""
        translator = TranslatorOperator(simulate_api=True)

        input_data = TranslateInput(
            text="Hello, world! This is a test of the translation system.",
            target_language="Spanish",
        )

        result = translator(inputs=input_data)

        assert isinstance(result, TranslateOutput)
        assert hasattr(result, "translated_text")
        assert isinstance(result.translated_text, str)
        assert hasattr(result, "detected_language")
        assert result.detected_language == "English"

    def test_sequential_pipeline(self):
        """Test a sequential pipeline of real operators with proper input mappings."""
        # Create the operators
        summarizer = SummarizerOperator(simulate_api=True)
        translator = TranslatorOperator(simulate_api=True)

        # Define input mappings between operators
        # For summarizer -> translator mapping:
        # - The summarizer output has "summary" and "word_count"
        # - The translator input needs "text" and "target_language"
        mapping = {
            "text": lambda result: result.summary,
            "target_language": lambda _: "French",  # Static value
        }

        # Create the pipeline with the mapping
        pipeline = SequentialPipeline(
            operators=[summarizer, translator], input_mappings=[mapping]
        )

        # Create the input for the summarizer only
        long_text = """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
        incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud 
        exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute 
        irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla 
        pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia 
        deserunt mollit anim id est laborum.
        """

        # Create just the input needed for the first operator
        input_data = SummarizeInput(text=long_text, max_words=20)

        # Execute the pipeline
        result = pipeline.run(input_data)

        # The result should be from the last operator (translator)
        assert isinstance(result, TranslateOutput)
        assert hasattr(result, "translated_text")
        assert isinstance(result.translated_text, str)

        # Since this is a simulated response, we can only check the structure
        assert len(result.translated_text) > 0

    def test_complex_operator_graph(self):
        """Test a more complex operator graph with multiple paths."""
        # Create the basic operators
        summarizer1 = SummarizerOperator(simulate_api=True)
        summarizer2 = SummarizerOperator(simulate_api=True)
        translator = TranslatorOperator(simulate_api=True)

        # Create the first pipeline - summarize then translate
        pipeline1 = SequentialPipeline(
            operators=[summarizer1, translator],
            input_mappings=[
                {
                    "text": lambda result: result.summary,
                    "target_language": lambda _: "Spanish",
                }
            ],
        )

        # Create the second pipeline - just summarize differently
        pipeline2 = SequentialPipeline(operators=[summarizer2])

        # Create a simple graph structure that executes both pipelines
        class SimpleGraph:
            """A simple graph implementation for testing."""

            def __init__(self, pipelines):
                self.pipelines = pipelines

            def execute(self, inputs):
                """Execute all pipelines and return a dictionary of results."""
                results = {}
                for i, pipeline in enumerate(self.pipelines):
                    results[f"branch_{i+1}"] = pipeline.run(inputs)
                return results

        # Create a complex graph with both pipelines
        graph = SimpleGraph([pipeline1, pipeline2])

        # Create input data
        long_text = """
        Science and technology have transformed the world in countless ways. From medical 
        advancements that extend human lifespans to communication tools that connect people 
        across vast distances instantly, innovation continues to reshape society. While these 
        developments bring tremendous benefits, they also present new challenges and ethical 
        considerations that require careful attention. As we look to the future, balancing 
        progress with responsibility will be essential for ensuring that technological development 
        serves humanity's best interests.
        """

        # Create the input for the graph
        input_data = SummarizeInput(text=long_text, max_words=25)

        # Execute the graph
        results = graph.execute(input_data)

        # Verify the results structure
        assert "branch_1" in results
        assert "branch_2" in results

        # Branch 1 should be a TranslateOutput (summarize + translate)
        branch1 = results["branch_1"]
        assert isinstance(branch1, TranslateOutput)
        assert hasattr(branch1, "translated_text")
        assert len(branch1.translated_text) > 0

        # Branch 2 should be a SummarizeOutput (just summarize)
        branch2 = results["branch_2"]
        assert isinstance(branch2, SummarizeOutput)
        assert hasattr(branch2, "summary")
        assert len(branch2.summary) > 0
