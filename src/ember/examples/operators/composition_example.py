"""Operator Composition with Enhanced JIT API.

This example demonstrates how to create complex pipelines by composing operators
with the enhanced JIT API. It shows three patterns:

1. Functional composition with the `compose` utility
2. Sequential operator chaining with explicit dependencies
3. Nested operators within a container class

All approaches benefit from automatic graph building and execution.

To run:
    poetry run python src/ember/examples/composition_example.py
"""

import logging
import time
from typing import Any, Dict, List, Callable, Optional, TypeVar, cast

from pydantic import BaseModel
from prettytable import PrettyTable

# ember API imports
from ember.api import non
from ember.api.operator import Operator, Specification
from ember.api.models import LMModule, LMModuleConfig
from ember.api.xcs import jit, execution_options

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


###############################################################################
# Composition Utilities
###############################################################################
def compose(f: Callable[[U], V], g: Callable[[T], U]) -> Callable[[T], V]:
    """Compose two functions: f ∘ g.

    Args:
        f: Function that takes output of g
        g: Function that takes initial input

    Returns:
        Composed function (f ∘ g)(x) = f(g(x))
    """

    def composed(x: T) -> V:
        return f(g(x))

    return composed


###############################################################################
# Custom Operators
###############################################################################
class QuestionRefinementInputs(BaseModel):
    """Input model for QuestionRefinement operator."""

    query: str


class QuestionRefinementOutputs(BaseModel):
    """Output model for QuestionRefinement operator."""

    refined_query: str


class QuestionRefinementSpecification(Specification):
    """Specification for QuestionRefinement operator."""

    input_model = QuestionRefinementInputs
    output_model = QuestionRefinementOutputs
    prompt_template = (
        "You are an expert at refining questions to make them clearer and more precise.\n"
        "Please refine the following question:\n\n"
        "{query}\n\n"
        "Provide a refined version that is more specific and answerable."
    )


@jit()
class QuestionRefinement(Operator[QuestionRefinementInputs, QuestionRefinementOutputs]):
    """Operator that refines a user question to make it more precise."""

    specification: ClassVar[Specification] = QuestionRefinementSpecification()
    model_name: str
    temperature: float

    def __init__(self, *, model_name: str, temperature: float = 0.3) -> None:
        self.model_name = model_name
        self.temperature = temperature

        # Configure internal LM module using the API
        self.lm_module = LMModule(
            config=LMModuleConfig(
                model_name=model_name,
                temperature=temperature,
            )
        )

    def forward(self, *, inputs: QuestionRefinementInputs) -> Dict[str, Any]:
        prompt = self.specification.render_prompt(inputs=inputs)
        response = self.lm_module(prompt=prompt)

        # Extract text from response
        from ember.api.types import extract_value

        refined_query = extract_value(response, "text", "").strip()

        return {"refined_query": refined_query}


###############################################################################
# Pipeline Pattern 1: Functional Composition
###############################################################################
def create_functional_pipeline(*, model_name: str) -> Callable[[Dict[str, Any]], Any]:
    """Create a pipeline using functional composition.

    Args:
        model_name: Name of the LLM to use

    Returns:
        A callable pipeline function
    """
    # Create individual operators
    refiner = QuestionRefinement(model_name=model_name)
    ensemble = non.UniformEnsemble(num_units=3, model_name=model_name, temperature=0.7)
    aggregator = non.MostCommon()

    # Use partial application to adapt the interfaces
    def adapt_refiner_output(inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = refiner(inputs=inputs)
        return {"query": result["refined_query"]}

    def adapt_ensemble_output(inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = ensemble(inputs=inputs)
        return {"query": inputs["query"], "responses": result["responses"]}

    # Compose the pipeline
    pipeline = compose(aggregator, compose(adapt_ensemble_output, adapt_refiner_output))

    return pipeline


###############################################################################
# Pipeline Pattern 2: Container Class with Nested Operators
###############################################################################
@jit(sample_input={"query": "What is the speed of light?"})
class NestedPipeline(Operator[Dict[str, Any], Dict[str, Any]]):
    """Pipeline implemented as a container class with nested operators."""

    def __init__(self, *, model_name: str) -> None:
        self.refiner = QuestionRefinement(model_name=model_name)
        self.ensemble = non.UniformEnsemble(
            num_units=3, model_name=model_name, temperature=0.7
        )
        self.aggregator = non.MostCommon()

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Step 1: Refine the question
        refined = self.refiner(inputs=inputs)

        # Step 2: Generate ensemble of answers
        ensemble_result = self.ensemble(inputs={"query": refined["refined_query"]})

        # Step 3: Aggregate results
        final_result = self.aggregator(
            inputs={
                "query": refined["refined_query"],
                "responses": ensemble_result["responses"],
            }
        )

        return final_result


###############################################################################
# Pipeline Pattern 3: Sequential Chaining
###############################################################################
def create_sequential_pipeline(*, model_name: str) -> Callable[[Dict[str, Any]], Any]:
    """Create a pipeline by explicitly chaining operators.

    Args:
        model_name: Name of the LLM to use

    Returns:
        A callable pipeline function
    """
    # Create individual operators
    refiner = QuestionRefinement(model_name=model_name)
    ensemble = non.UniformEnsemble(num_units=3, model_name=model_name, temperature=0.7)
    aggregator = non.MostCommon()

    # Create the chained function
    def pipeline(inputs: Dict[str, Any]) -> Any:
        refined = refiner(inputs=inputs)
        ensemble_result = ensemble(inputs={"query": refined["refined_query"]})
        final_result = aggregator(
            inputs={
                "query": refined["refined_query"],
                "responses": ensemble_result["responses"],
            }
        )
        return final_result

    return pipeline


###############################################################################
# Main Demonstration
###############################################################################
def main() -> None:
    """Run demonstration of different composition patterns."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Define configuration parameters
    model_name: str = "openai:gpt-4o-mini"

    # Create pipelines using different patterns
    functional_pipeline = create_functional_pipeline(model_name=model_name)
    nested_pipeline = NestedPipeline(model_name=model_name)
    sequential_pipeline = create_sequential_pipeline(model_name=model_name)

    # List of questions to process
    questions: List[str] = [
        "How does gravity work?",
        "Tell me about the history of Rome",
        "What's the difference between DNA and RNA?",
    ]

    # Prepare table for results comparison
    table = PrettyTable()
    table.field_names = ["Pipeline", "Time (s)", "Result"]
    table.align = "l"

    # Process questions with each pipeline
    print("\n=== Functional Composition Pipeline ===")
    for question in questions[:1]:  # Use first question only for brevity
        print(f"\nProcessing: {question}")
        start_time = time.perf_counter()
        result = functional_pipeline({"query": question})
        elapsed = time.perf_counter() - start_time

        # Show details of pipeline execution
        print(f'Original query: "{question}"')
        if "refined_query" in result:
            print(f"Refined query: \"{result['refined_query']}\"")
        print(
            f"Final answer: \"{result['final_answer'][:150]}...\""
            if len(result["final_answer"]) > 150
            else f"Final answer: \"{result['final_answer']}\""
        )
        print(f"Time: {elapsed:.4f}s")

        # Store in table
        table.add_row(
            [
                "Functional",
                f"{elapsed:.4f}",
                result["final_answer"][:50] + "..."
                if len(result["final_answer"]) > 50
                else result["final_answer"],
            ]
        )

    print("\n=== Nested Pipeline ===")
    for question in questions[:1]:
        print(f"\nProcessing: {question}")
        start_time = time.perf_counter()
        result = nested_pipeline(inputs={"query": question})
        elapsed = time.perf_counter() - start_time

        # Show details of pipeline execution
        print(
            f"Final answer: \"{result['final_answer'][:150]}...\""
            if len(result["final_answer"]) > 150
            else f"Final answer: \"{result['final_answer']}\""
        )
        print(f"Time: {elapsed:.4f}s")

        # Store in table
        table.add_row(
            [
                "Nested",
                f"{elapsed:.4f}",
                result["final_answer"][:50] + "..."
                if len(result["final_answer"]) > 50
                else result["final_answer"],
            ]
        )

    print("\n=== Sequential Pipeline ===")
    for question in questions[:1]:
        print(f"\nProcessing: {question}")
        start_time = time.perf_counter()
        result = sequential_pipeline({"query": question})
        elapsed = time.perf_counter() - start_time

        # Show details of pipeline execution
        print(
            f"Final answer: \"{result['final_answer'][:150]}...\""
            if len(result["final_answer"]) > 150
            else f"Final answer: \"{result['final_answer']}\""
        )
        print(f"Time: {elapsed:.4f}s")

        # Store in table
        table.add_row(
            [
                "Sequential",
                f"{elapsed:.4f}",
                result["final_answer"][:50] + "..."
                if len(result["final_answer"]) > 50
                else result["final_answer"],
            ]
        )

    # Demonstrate execution options with the nested pipeline
    print("\n=== Nested Pipeline with Sequential Execution ===")
    with execution_options(scheduler="sequential"):
        for question in questions[:1]:
            print(f"\nProcessing: {question}")
            start_time = time.perf_counter()
            result = nested_pipeline(inputs={"query": question})
            elapsed = time.perf_counter() - start_time

            # Show details of pipeline execution
            print(
                f"Final answer: \"{result['final_answer'][:150]}...\""
                if len(result["final_answer"]) > 150
                else f"Final answer: \"{result['final_answer']}\""
            )
            print(f"Time: {elapsed:.4f}s")
            print(f"Execution mode: Sequential scheduler")

            # Store in table
            table.add_row(
                [
                    "Nested (Sequential)",
                    f"{elapsed:.4f}",
                    result["final_answer"][:50] + "..."
                    if len(result["final_answer"]) > 50
                    else result["final_answer"],
                ]
            )

    # Display performance comparison
    print("\n=== Performance Comparison ===")
    print(table)

    print("\n=== Pipeline Pattern Comparison ===")
    print("1. Functional Composition:")
    print("   • Advantages: Clean separation of concerns, explicit data flow")
    print("   • Use cases: When components need to be reused separately")

    print("\n2. Nested Pipeline:")
    print("   • Advantages: Benefits from JIT optimization, cleaner code structure")
    print("   • Use cases: Complex pipelines where performance matters")

    print("\n3. Sequential Pipeline:")
    print("   • Advantages: Simplicity, flexibility for custom logic")
    print("   • Use cases: Prototyping or simpler workflows")


if __name__ == "__main__":
    main()
