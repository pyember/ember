"""Setup-wizard config bridge."""

import json
import sys
import tempfile
from pathlib import Path

from ember._internal.context import EmberContext


def _validate_provider(provider: str) -> str:
    if not provider or not provider.replace("_", "").isalnum():
        raise ValueError(f"Invalid provider name: {provider}")
    return provider


def _validate_api_key(api_key: str) -> str:
    api_key = api_key.strip()
    if not api_key:
        raise ValueError("API key cannot be empty")
    if len(api_key) < 5:
        raise ValueError("API key appears to be too short")
    if api_key.startswith('"') and api_key.endswith('"'):
        raise ValueError("API key should not be quoted")
    if " " in api_key:
        raise ValueError("API key should not contain spaces")
    return api_key


def save_api_key(provider: str, api_key: str) -> bool:
    """Save API key to providers.<provider>.api_key in config.yaml.

    Args:
        provider: Provider name (e.g., "openai")
        api_key: API key to save

    Returns:
        True if successful

    Raises:
        ValueError: Invalid provider name or API key
    """
    provider = _validate_provider(provider)
    api_key = _validate_api_key(api_key)

    ctx = EmberContext.current()
    ctx.set_config(f"providers.{provider}.api_key", api_key)
    ctx.save()
    return True


def save_config(config_updates: dict[str, object]) -> None:
    """Merge configuration updates into the current context."""

    def _apply_nested(ctx: EmberContext, updates: dict[str, object], prefix: str = "") -> None:
        for key, value in updates.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                _apply_nested(ctx, dict(value), full_key)
            else:
                ctx.set_config(full_key, value)

    ctx = EmberContext.current()
    _apply_nested(ctx, config_updates)
    ctx.save()


def test_api_key(provider: str, api_key: str, model: str) -> str:
    """Validate an API key by invoking a simple model request.

    The test runs inside a temporary, isolated EmberContext so credentials are
    never persisted to the user's config during validation.
    """
    provider = _validate_provider(provider)
    api_key = _validate_api_key(api_key)

    with tempfile.TemporaryDirectory(prefix="ember-setup-test-") as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        config_path.write_text("{}", encoding="utf-8")

        ctx = EmberContext(config_path=config_path, isolated=True)
        ctx.set_config(f"providers.{provider}.api_key", api_key)

        from ember.api import models as models_api

        with ctx:
            response = models_api(model, "Say hello!")
            return str(response)


def main() -> None:
    """CLI interface for configuration API.

    Reads sensitive data from stdin to avoid exposing in process lists.
    """
    if len(sys.argv) < 2:
        print(
            "Usage: python -m ember.cli.commands.configure_api <command> [args]",
            file=sys.stderr,
        )
        sys.exit(1)

    command = sys.argv[1]

    try:
        if command == "save-key":
            if len(sys.argv) != 3:
                print("Usage: configure_api save-key <provider>", file=sys.stderr)
                print("API key should be provided via stdin", file=sys.stderr)
                sys.exit(1)

            provider = sys.argv[2]
            api_key = sys.stdin.read().strip()

            if not api_key:
                print("No API key provided", file=sys.stderr)
                sys.exit(1)

            save_api_key(provider, api_key)

        elif command == "test-key":
            if len(sys.argv) != 4:
                print("Usage: configure_api test-key <provider> <model>", file=sys.stderr)
                print("API key should be provided via stdin", file=sys.stderr)
                sys.exit(1)

            provider = sys.argv[2]
            model = sys.argv[3]
            api_key = sys.stdin.read().strip()

            if not api_key:
                print("No API key provided", file=sys.stderr)
                sys.exit(1)

            output = test_api_key(provider, api_key, model)
            print(f"SUCCESS:{output}")

        elif command == "save-config":
            config_json = sys.stdin.read().strip()

            if not config_json:
                print("No configuration provided", file=sys.stderr)
                sys.exit(1)

            config = json.loads(config_json)
            if not isinstance(config, dict):
                raise ValueError("Configuration payload must be a JSON object")
            save_config(config)

        else:
            print(f"Unknown command: {command}", file=sys.stderr)
            sys.exit(1)

    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
