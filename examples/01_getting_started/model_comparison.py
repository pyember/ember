"""Compare responses from different language models.

This example demonstrates:
- Calling multiple models with the same prompt
- Comparing response quality and characteristics
- Using different providers (OpenAI, Anthropic, Google)

Prerequisites:
    Configure providers you want to compare:
        ember configure set providers.openai.api_key "sk-..."
        ember configure set providers.anthropic.api_key "sk-ant-..."
        ember configure set providers.google.api_key "..."

Run with:
    python examples/01_getting_started/model_comparison.py
"""

from __future__ import annotations

from ember.api import models


# Models to compare (you may not have all providers configured)
MODELS_TO_COMPARE = [
    "gpt-4o-mini",
    "claude-3-5-haiku-latest",
    "gemini-2.0-flash",
]


def compare_models(prompt: str) -> None:
    """Send the same prompt to multiple models and compare responses."""
    print(f"Prompt: {prompt}")
    print("=" * 60)
    print()

    for model_id in MODELS_TO_COMPARE:
        print(f"Model: {model_id}")
        print("-" * 40)

        try:
            response = models.response(model_id, prompt)
            print(f"Response: {response.text[:200]}...")
            usage = response.usage
            print(f"Tokens: {usage.get('total_tokens', 'N/A')}")
        except Exception as e:
            error_msg = str(e)
            if "API key" in error_msg or "credentials" in error_msg.lower():
                print(f"Skipped: API key not configured for this provider")
            else:
                print(f"Error: {error_msg[:100]}")

        print()


def main() -> None:
    """Run the model comparison example."""
    print("Model Comparison Example")
    print("=" * 60)
    print()

    # Compare on a simple task
    prompt = (
        "Explain the concept of recursion in programming. "
        "Use a simple example and keep your response under 100 words."
    )
    compare_models(prompt)

    print("Key observations:")
    print("  - Different models have different response styles")
    print("  - Token usage varies by model")
    print("  - Some models are faster than others")
    print()
    print("Next: Try examples/01_getting_started/basic_prompt_engineering.py")


if __name__ == "__main__":
    main()
