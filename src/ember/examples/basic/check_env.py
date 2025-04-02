"""Environment Variable Check

This script checks if API keys are properly set in the environment variables.

To run:
    uv run python src/ember/examples/basic/check_env.py
"""

import os


def main():
    """Print environment variables related to API keys."""
    print("\n=== Environment Variables Check ===\n")

    # Check for OpenAI API key
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        print(f"OPENAI_API_KEY: {'*' * (len(openai_key) - 4) + openai_key[-4:]}")
    else:
        print("OPENAI_API_KEY: Not set")

    # Check for Anthropic API key
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        print(
            f"ANTHROPIC_API_KEY: {'*' * (len(anthropic_key) - 4) + anthropic_key[-4:]}"
        )
    else:
        print("ANTHROPIC_API_KEY: Not set")

    print("\nAll environment variables:")
    for key, value in os.environ.items():
        # Only print sensitive values with asterisks
        if "KEY" in key or "SECRET" in key or "TOKEN" in key or "PASSWORD" in key:
            masked_value = "****" if value else "Not set"
            print(f"  {key}: {masked_value}")
        elif (
            "PATH" not in key
            and "SHELL" not in key
            and "EDITOR" not in key
            and "TERM" not in key
        ):
            # Skip common environment variables like PATH, SHELL, etc.
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
