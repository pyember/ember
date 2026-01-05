"""Basic prompt engineering techniques with Ember.

This example demonstrates:
- Using system prompts to set context
- Controlling output format
- Temperature and sampling parameters
- Few-shot prompting patterns

Prerequisites:
    Configure at least one provider:
        ember configure set providers.openai.api_key "sk-..."

Run with:
    python examples/01_getting_started/basic_prompt_engineering.py
"""

from __future__ import annotations

from ember.api import models


def demonstrate_system_prompts() -> None:
    """Show how system prompts affect responses."""
    print("System Prompts")
    print("-" * 40)

    user_message = "What's the best programming language?"

    # Without system prompt
    print("Without system prompt:")
    try:
        response = models("gpt-4o-mini", user_message)
        print(f"  {response[:150]}...")
    except Exception as e:
        print(f"  Error: {e}")
        return

    print()

    # With system prompt for conciseness
    print("With system prompt (concise expert):")
    response = models(
        "gpt-4o-mini",
        user_message,
        system="You are a concise programming expert. Answer in one sentence.",
    )
    print(f"  {response}")
    print()


def demonstrate_temperature() -> None:
    """Show how temperature affects response creativity."""
    print("Temperature Control")
    print("-" * 40)

    prompt = "Generate a creative name for a coffee shop."

    # Low temperature (more deterministic)
    print("Low temperature (0.1) - consistent:")
    for i in range(3):
        try:
            response = models("gpt-4o-mini", prompt, temperature=0.1)
            print(f"  Attempt {i+1}: {response}")
        except Exception as e:
            print(f"  Error: {e}")
            return
    print()

    # High temperature (more creative)
    print("High temperature (1.0) - varied:")
    for i in range(3):
        response = models("gpt-4o-mini", prompt, temperature=1.0)
        print(f"  Attempt {i+1}: {response}")
    print()


def demonstrate_structured_output() -> None:
    """Show how to get structured responses."""
    print("Structured Output")
    print("-" * 40)

    prompt = """Analyze this text and respond in JSON format:

    Text: "The quick brown fox jumps over the lazy dog."

    Format: {"word_count": N, "longest_word": "...", "has_animals": true/false}
    """

    try:
        response = models(
            "gpt-4o-mini",
            prompt,
            system="You are a JSON-only responder. Output valid JSON, nothing else.",
            temperature=0.0,
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    print()


def main() -> None:
    """Run the prompt engineering example."""
    print("Basic Prompt Engineering")
    print("=" * 40)
    print()

    demonstrate_system_prompts()
    demonstrate_temperature()
    demonstrate_structured_output()

    print("Key takeaways:")
    print("  - System prompts set the model's persona and constraints")
    print("  - Temperature controls creativity vs consistency")
    print("  - Clear formatting instructions improve structured output")
    print()
    print("Next: Explore examples/02_core_concepts/ for operators and composition")


if __name__ == "__main__":
    main()
