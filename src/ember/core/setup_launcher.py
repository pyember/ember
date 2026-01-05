"""Minimal launcher for the setup wizard."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from ember._internal.context import EmberContext
from ember.core.credentials import CredentialNotFoundError


def launch_setup_if_needed(provider: str, model_id: str) -> str | None:
    """Launch setup wizard if in interactive mode.

    Args:
        provider: Provider name (e.g., "openai")
        model_id: Model being accessed

    Returns:
        API key if setup succeeds, None otherwise.
    """
    # Check if we're in an interactive terminal
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None

    # Check if npx is available
    if not shutil.which("npx"):
        # Fall back to simple text prompt
        return _simple_prompt(provider, model_id)

    # Show the Codex-style prompt
    print(f"\n{model_id} requires an API key from {provider.title()}.")
    print("Sign in to get an API key or paste one you already have.")
    print("\033[90m[use arrows to move, enter to select]\033[0m\n")

    # Launch the npm wizard with specific context
    try:
        env = os.environ.copy()
        env["EMBER_CONFIG_PATH"] = str(EmberContext.get_config_path())
        env["EMBER_SETUP_PROVIDER"] = provider
        env["EMBER_SETUP_MODEL"] = model_id
        env["EMBER_SETUP_CONTEXT"] = "missing-key"

        result = subprocess.run(["npx", "-y", "@ember-ai/setup"], env=env, capture_output=False)

        if result.returncode == 0:
            # Check if key is now available through context
            ctx = EmberContext.current()
            ctx.reload()
            try:
                api_key = ctx.get_credential(provider)
                return api_key
            except CredentialNotFoundError:
                return None

    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"\n\nSetup wizard failed: {exc}")
        return _simple_prompt(provider, model_id)

    return None


def _simple_prompt(provider: str, model_id: str) -> str | None:
    """Simple fallback prompt when npm is unavailable."""
    print(f"\n{model_id} requires an API key.")
    print(f"Enter your {provider.title()} API key: ", end="", flush=True)

    api_key = input().strip()

    if api_key:
        ctx = EmberContext.current()
        ctx.set_config(f"providers.{provider}.api_key", api_key)
        ctx.save()
        print(f"\nSaved API key to {EmberContext.get_config_path()}.")
        return api_key

    return None


def format_non_interactive_error(provider: str, model_id: str) -> str:
    """Format error for non-interactive environments."""
    urls = {
        "openai": "https://platform.openai.com/api-keys",
        "anthropic": "https://console.anthropic.com/api-keys",
        "google": "https://makersuite.google.com/app/apikey",
    }

    url = urls.get(provider, f"https://{provider}.com")

    return (
        f"No API key found for {model_id}.\n\n"
        f"To fix this, choose one:\n\n"
        f"Option 1: Run interactive setup (recommended)\n"
        f"   ember setup\n\n"
        f"Option 2: Save to config\n"
        f"   ember configure set providers.{provider}.api_key YOUR_KEY\n\n"
        f"Get your API key from: {url}"
    )
