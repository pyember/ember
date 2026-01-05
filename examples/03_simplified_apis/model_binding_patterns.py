"""Patterns for binding and configuring models.

This example demonstrates:
- Model discovery and listing
- Model binding for reuse
- Configuration patterns
- Multi-model strategies

Prerequisites:
    For LLM examples: ember configure set providers.openai.api_key "sk-..."

Run with:
    python examples/03_simplified_apis/model_binding_patterns.py
"""

from __future__ import annotations

from functools import partial
from typing import Callable

from ember.api import Models, models, op


# =============================================================================
# Part 1: Model Discovery
# =============================================================================

def demonstrate_discovery() -> None:
    """Show model discovery capabilities."""
    print("Part 1: Model Discovery")
    print("-" * 50)

    # List all available models
    all_models = models.list()
    print(f"Total models available: {len(all_models)}")
    print(f"Sample: {all_models[:5]}")
    print()

    # List providers
    providers = models.providers()
    print(f"Available providers: {providers}")
    print()

    # Using model constants for safety
    print("Model constants (prevents typos):")
    print(f"  Models.GPT_4 = {Models.GPT_4!r}")
    print(f"  Models.GPT_4O_MINI = {Models.GPT_4O_MINI!r}")
    print()


# =============================================================================
# Part 2: Model Binding Patterns
# =============================================================================

def create_model_caller(model_id: str, **defaults) -> Callable[[str], str]:
    """Create a bound model caller with defaults.

    This pattern is useful when you want to reuse the same
    model configuration across multiple calls.
    """

    @op
    def call_model(prompt: str) -> str:
        try:
            return models(model_id, prompt, **defaults)
        except Exception as e:
            return f"[Error: {e}]"

    return call_model


# Pre-configured model callers
fast_model = create_model_caller("gpt-4o-mini", temperature=0.3)
creative_model = create_model_caller("gpt-4o-mini", temperature=1.0)


# =============================================================================
# Part 3: Model Selection Strategies
# =============================================================================

class ModelRouter:
    """Route requests to appropriate models based on criteria."""

    def __init__(self):
        self._routes: dict[str, str] = {
            "fast": "gpt-4o-mini",
            "quality": "gpt-4o",
            "creative": "gpt-4o-mini",
            "code": "gpt-4o",
        }

    def get_model(self, task_type: str) -> str:
        """Get the appropriate model for a task type."""
        return self._routes.get(task_type, "gpt-4o-mini")

    def route_and_call(self, task_type: str, prompt: str) -> dict:
        """Route to appropriate model and make the call."""
        model_id = self.get_model(task_type)
        return {
            "task_type": task_type,
            "model": model_id,
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "note": "Would call model here with API key",
        }


# =============================================================================
# Part 4: Configuration Patterns
# =============================================================================

class ModelConfig:
    """Encapsulate model configuration."""

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def to_dict(self) -> dict:
        """Convert to kwargs for model call."""
        result = {"temperature": self.temperature}
        if self.max_tokens:
            result["max_tokens"] = self.max_tokens
        if self.system_prompt:
            result["system"] = self.system_prompt
        return result

    def __repr__(self) -> str:
        return f"ModelConfig({self.model_id}, temp={self.temperature})"


# Pre-defined configurations
CONFIGS = {
    "precise": ModelConfig("gpt-4o", temperature=0.1, max_tokens=500),
    "creative": ModelConfig("gpt-4o-mini", temperature=0.9, max_tokens=1000),
    "concise": ModelConfig(
        "gpt-4o-mini",
        temperature=0.3,
        max_tokens=100,
        system_prompt="Be concise. Answer in one sentence.",
    ),
}


def demonstrate_configs() -> None:
    """Show configuration patterns."""
    print("Part 4: Configuration Patterns")
    print("-" * 50)

    for name, config in CONFIGS.items():
        print(f"  {name}: {config}")
        print(f"    kwargs: {config.to_dict()}")
    print()


# =============================================================================
# Part 5: Partial Application Pattern
# =============================================================================

def generic_summarizer(text: str, model_id: str, style: str = "brief") -> dict:
    """Summarize text with configurable model and style.

    Use functools.partial to create specialized versions.
    """
    styles = {
        "brief": "Summarize in one sentence.",
        "detailed": "Provide a detailed summary with key points.",
        "bullet": "Summarize as bullet points.",
    }

    return {
        "model": model_id,
        "style": style,
        "instruction": styles.get(style, styles["brief"]),
        "text_preview": text[:50] + "..." if len(text) > 50 else text,
    }


# Create specialized summarizers using partial
brief_summarize = partial(generic_summarizer, model_id="gpt-4o-mini", style="brief")
detailed_summarize = partial(generic_summarizer, model_id="gpt-4o", style="detailed")


def main() -> None:
    """Demonstrate model binding patterns."""
    print("Model Binding Patterns")
    print("=" * 50)
    print()

    # Part 1: Discovery
    demonstrate_discovery()

    # Part 2: Bound callers
    print("Part 2: Model Binding")
    print("-" * 50)
    print("Created bound callers:")
    print("  fast_model: gpt-4o-mini, temp=0.3")
    print("  creative_model: gpt-4o-mini, temp=1.0")
    print()

    # Part 3: Routing
    print("Part 3: Model Routing")
    print("-" * 50)
    router = ModelRouter()
    for task in ["fast", "quality", "creative", "code"]:
        model = router.get_model(task)
        print(f"  {task:10} -> {model}")
    print()

    # Part 4: Configs
    demonstrate_configs()

    # Part 5: Partial application
    print("Part 5: Partial Application")
    print("-" * 50)
    sample = "This is a long document that needs summarization..."
    brief = brief_summarize(sample)
    detailed = detailed_summarize(sample)
    print(f"Brief:    {brief['model']}, {brief['style']}")
    print(f"Detailed: {detailed['model']}, {detailed['style']}")
    print()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Use models.list() and models.providers() for discovery")
    print("2. Bind models with defaults using closures or partial")
    print("3. Route requests to models based on task requirements")
    print("4. Encapsulate configuration in reusable objects")
    print()
    print("Next: Explore examples/04_compound_ai/ for multi-model systems")


if __name__ == "__main__":
    main()
