"""
Advanced Benchmark: Measuring Parallelization Impact in Ensemble LLM Execution

This dedicated benchmark script measures the performance difference between:
1. Sequential execution
2. Auto-parallelized execution
3. Explicitly parallelized execution

It uses the ensemble pattern from ensemble_judge_mmlu.py but focuses purely on
measuring execution strategies with precise instrumentation.

Usage:
    MODEL_COUNT=10 uv run python -m ember.examples.advanced.parallel_benchmark

Environment variables:
    MODEL_COUNT: Number of models to use in the ensemble (default: 10)
    REPEAT: Number of benchmark repetitions (default: 3)
"""

import time
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, cast

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import necessary modules
from ember.api import models, operators
from ember.api.operators import Operator
from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig
from ember.core.registry.operator.base.operator_base import Specification
from ember.core.registry.specification.specification import (
    Specification as CoreSpecification,
)
from ember.core.types.ember_model import EmberModel
from ember.xcs import jit, pmap
from ember.xcs.engine.execution_options import execution_options

# Set up console for rich output
console = Console()


# Define input/output data structures using EmberModel for compatibility with XCS
class BenchmarkInput(EmberModel):
    """Input for benchmark."""

    prompt: str


class BenchmarkOutput(EmberModel):
    """Output for benchmark."""

    result: str
    model_name: str


class BenchmarkEnsembleOutput(EmberModel):
    """Output from ensemble."""

    results: List[BenchmarkOutput]


class BenchmarkSpecification(CoreSpecification):
    """Specification for benchmark operator."""

    # Define both input and output models explicitly
    input_model: Type[BenchmarkInput] = BenchmarkInput
    structured_output: Type[BenchmarkOutput] = BenchmarkOutput

    prompt_template: str = """Respond with the exact phrase: 'This is a benchmark response from {model_name}' 
    
Here is the input prompt: {prompt}

Your response must contain only the requested phrase.
"""


class SingleModelOperator(Operator[BenchmarkInput, BenchmarkOutput]):
    """Baseline operator using a single model for benchmarking."""

    specification: ClassVar[Specification] = BenchmarkSpecification()
    model_name: str
    temperature: float
    max_tokens: int
    lm_module: LMModule

    def __init__(
        self,
        model_name: str = "anthropic:claude-3-haiku",
        temperature: float = 0.0,
        max_tokens: int = 16,
    ) -> None:
        """Initialize the benchmark operator with model configuration."""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize the LM module
        self.lm_module = LMModule(
            config=LMModuleConfig(
                model_name=model_name, temperature=temperature, max_tokens=max_tokens
            )
        )

    def forward(self, *, inputs: BenchmarkInput) -> BenchmarkOutput:
        """Process a prompt with the model."""
        # Create template variables with the model name included
        template_vars = {"prompt": inputs.prompt, "model_name": self.model_name}

        # Fill in the template with standard Python format strings
        prompt = self.specification.prompt_template.format(**template_vars)

        # Get model response
        response = self.lm_module(prompt=prompt)

        return BenchmarkOutput(result=response, model_name=self.model_name)


class EnsembleSpecification(CoreSpecification):
    """Specification for ensemble operator."""

    # Explicit models for proper type handling
    input_model: Type[BenchmarkInput] = BenchmarkInput
    structured_output: Type[BenchmarkEnsembleOutput] = BenchmarkEnsembleOutput


@jit
class EnsembleOperator(Operator[BenchmarkInput, BenchmarkEnsembleOutput]):
    """JIT-optimized parallel ensemble operator.

    This operator runs multiple model workers in parallel using the same input.
    The JIT optimization automatically parallelizes the model calls when possible.
    """

    specification: ClassVar[Specification] = EnsembleSpecification()
    models: List[SingleModelOperator]

    def __init__(self, model_configs: List[Dict[str, Any]]) -> None:
        """Initialize the ensemble with multiple model operators."""
        self.models = []
        for config in model_configs:
            self.models.append(
                SingleModelOperator(
                    model_name=config["model_name"],
                    temperature=config.get("temperature", 0.0),
                    max_tokens=config.get("max_tokens", 16),
                )
            )

    def forward(self, *, inputs: BenchmarkInput) -> BenchmarkEnsembleOutput:
        """Process the input through all models in parallel."""
        results = []

        # Process with each model - the JIT optimizer can recognize these as independent operations
        for model_op in self.models:
            # Each call can be parallelized
            result = model_op(inputs=inputs)
            results.append(result)

        return BenchmarkEnsembleOutput(results=results)


def run_benchmarks(
    model_configs: List[Dict[str, Any]], repeats: int = 3
) -> Dict[str, Any]:
    """Run benchmarks comparing sequential vs parallel execution.

    Performs multiple timing runs to ensure statistical significance.

    Args:
        model_configs: List of model configurations to use
        repeats: Number of benchmark repetitions

    Returns:
        Dictionary of benchmark results
    """
    # Create test input
    test_input = BenchmarkInput(prompt="What are compound AI systems?")

    # Create ensemble operator
    ensemble = EnsembleOperator(model_configs)

    # Statistics for each approach
    sequential_times = []
    auto_parallel_times = []
    explicit_parallel_times = []

    # Print benchmark configuration
    num_models = len(model_configs)
    console.print(
        f"[bold]Running benchmark with {num_models} models, {repeats} repetitions[/bold]"
    )
    model_table = Table(title="Models Used")
    model_table.add_column("Index", style="cyan")
    model_table.add_column("Model", style="green")
    model_table.add_column("Temperature", style="yellow")

    for i, config in enumerate(model_configs):
        model_table.add_row(
            str(i + 1), config["model_name"], str(config.get("temperature", 0.0))
        )
    console.print(model_table)

    # Run benchmarks
    console.print("\n[bold]Starting benchmark runs...[/bold]")

    for i in range(repeats):
        console.print(f"\n[bold]Benchmark Run {i+1}/{repeats}[/bold]")

        # 1. Sequential execution (force sequential with execution_options)
        console.print("  Measuring sequential execution...")
        start_time = time.perf_counter()
        with execution_options(use_parallel=False):
            sequential_result = ensemble(inputs=test_input)
        sequential_time = time.perf_counter() - start_time
        sequential_times.append(sequential_time)
        console.print(f"    Sequential time: {sequential_time:.4f}s")

        # 2. Auto-parallel execution (default JIT behavior)
        console.print("  Measuring auto-parallel execution...")
        start_time = time.perf_counter()
        auto_result = ensemble(inputs=test_input)
        auto_time = time.perf_counter() - start_time
        auto_parallel_times.append(auto_time)
        console.print(f"    Auto-parallel time: {auto_time:.4f}s")

        # 3. Explicit parallel execution
        console.print("  Measuring explicit parallel execution...")
        start_time = time.perf_counter()
        with execution_options(use_parallel=True):
            explicit_result = ensemble(inputs=test_input)
        explicit_time = time.perf_counter() - start_time
        explicit_parallel_times.append(explicit_time)
        console.print(f"    Explicit parallel time: {explicit_time:.4f}s")

    # Calculate averages
    avg_sequential = sum(sequential_times) / len(sequential_times)
    avg_auto = sum(auto_parallel_times) / len(auto_parallel_times)
    avg_explicit = sum(explicit_parallel_times) / len(explicit_parallel_times)

    # Calculate speedups
    auto_speedup = avg_sequential / avg_auto if avg_auto > 0 else 0
    explicit_speedup = avg_sequential / avg_explicit if avg_explicit > 0 else 0

    # Calculate time reductions as percentages
    auto_reduction = (
        ((avg_sequential - avg_auto) / avg_sequential) * 100
        if avg_sequential > 0
        else 0
    )
    explicit_reduction = (
        ((avg_sequential - avg_explicit) / avg_sequential) * 100
        if avg_sequential > 0
        else 0
    )

    # Print clean, elegant results
    console.print("\n[bold]Benchmark Results[/bold]")

    results_table = Table(
        title=f"JIT Optimization Performance ({len(model_configs)} models)"
    )
    results_table.add_column("Execution Strategy", style="cyan")
    results_table.add_column("Avg Time", style="yellow")
    results_table.add_column("Speedup", style="green")

    results_table.add_row("Sequential", f"{avg_sequential:.4f}s", "1.00x (baseline)")

    results_table.add_row("Auto Parallel", f"{avg_auto:.4f}s", f"{auto_speedup:.2f}x")

    results_table.add_row(
        "Explicit Parallel", f"{avg_explicit:.4f}s", f"{explicit_speedup:.2f}x"
    )

    console.print(results_table)

    # Show which strategy performed best, in a simple way
    best_strategy = (
        "Auto Parallel"
        if auto_speedup > explicit_speedup
        else "Explicit Parallel"
        if explicit_speedup > auto_speedup
        else "Equal Performance"
    )
    console.print(f"\n[bold]Best strategy:[/bold] {best_strategy}")

    # Show raw run data
    run_table = Table(title="Individual Run Times")
    run_table.add_column("Run", style="cyan")
    run_table.add_column("Sequential", style="red")
    run_table.add_column("Auto Parallel", style="yellow")
    run_table.add_column("Explicit Parallel", style="green")

    for i in range(len(sequential_times)):
        run_table.add_row(
            f"Run {i+1}",
            f"{sequential_times[i]:.4f}s",
            f"{auto_parallel_times[i]:.4f}s",
            f"{explicit_parallel_times[i]:.4f}s",
        )

    console.print(run_table)

    # Return detailed results for further analysis
    return {
        "sequential_times": sequential_times,
        "auto_parallel_times": auto_parallel_times,
        "explicit_parallel_times": explicit_parallel_times,
        "avg_sequential": avg_sequential,
        "avg_auto": avg_auto,
        "avg_explicit": avg_explicit,
        "auto_speedup": auto_speedup,
        "explicit_speedup": explicit_speedup,
        "auto_reduction": auto_reduction,
        "explicit_reduction": explicit_reduction,
        "num_models": len(model_configs),
    }


def main() -> None:
    """Main function to run the parallelization benchmark."""
    import os

    # Print header
    console.print(
        Panel.fit(
            "Measuring JIT Parallelization Impact in Ensemble LLM Execution",
            title="Ember Parallelization Benchmark",
            subtitle="Comparing Sequential vs Auto-Parallel vs Explicit Parallel",
            style="bold cyan",
        )
    )

    # Check if API keys are available
    api_keys = {
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
    }

    if not any(api_keys.values()):
        console.print(
            Panel(
                "⚠️ No API keys found for language model providers.\n\n"
                "To run this benchmark, please set at least one of these environment variables:\n"
                "  - ANTHROPIC_API_KEY\n"
                "  - OPENAI_API_KEY\n"
                "  - GOOGLE_API_KEY\n\n"
                "Example:\n"
                "  export ANTHROPIC_API_KEY=sk_ant_xxxx",
                title="API Keys Required",
                style="yellow",
            )
        )
        return

    # Basic model configurations - prefer lightweight models for benchmarking
    model_configs = [
        {
            "model_name": "anthropic:claude-3-haiku",
            "temperature": 0.0,
            "max_tokens": 16,
        },
        {
            "model_name": "anthropic:claude-3-haiku",
            "temperature": 0.7,
            "max_tokens": 16,
        },
    ]

    # Add OpenAI model if available
    if os.environ.get("OPENAI_API_KEY"):
        model_configs.append(
            {"model_name": "openai:gpt-3.5-turbo", "temperature": 0.0, "max_tokens": 16}
        )

    # Get number of models to use
    model_count = min(int(os.environ.get("MODEL_COUNT", "2")), len(model_configs))
    repeats = int(os.environ.get("REPEAT", "3"))

    try:
        # Run benchmarks
        results = run_benchmarks(
            model_configs=model_configs[:model_count], repeats=repeats
        )

        # Final summary
        console.print("\n[bold green]Benchmark completed successfully![/bold green]")
        best_speedup = max(results["auto_speedup"], results["explicit_speedup"])
        console.print(
            f"[bold]JIT parallelization achieved {best_speedup:.2f}x speedup[/bold]"
        )

    except Exception as e:
        console.print(
            Panel(
                f"Error running benchmark: {str(e)}\n\n"
                "This benchmark requires properly configured API keys and model availability.\n"
                "Please check your API keys and available models.",
                title="Execution Error",
                style="red",
            )
        )


if __name__ == "__main__":
    main()
