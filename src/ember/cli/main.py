"""Ember CLI entrypoint."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ember._internal.context import EmberContext
from ember.cli.commands.configure import cmd_configure
from ember.cli.commands.context import cmd_context_validate, cmd_context_view
from ember.cli.commands.registry import (
    cmd_registry_info,
    cmd_registry_list_models,
    cmd_registry_list_providers,
)


def cmd_setup(args: argparse.Namespace) -> int:
    ctx = args.context

    if not shutil.which("npx"):
        print("Error: npm/npx is required for the setup wizard.")
        print("Install Node.js from https://nodejs.org/")
        return 1

    import os

    env = os.environ.copy()
    env["EMBER_CONFIG_PATH"] = str(EmberContext.get_config_path())

    try:
        setup_wizard_path = Path(__file__).parent / "setup-wizard"
        if setup_wizard_path.exists():
            build_result = subprocess.run(
                ["npm", "run", "build"], cwd=setup_wizard_path, capture_output=True
            )
            if build_result.returncode == 0:
                result = subprocess.run(["npm", "run", "start"], cwd=setup_wizard_path, env=env)
            else:
                print("Error building setup wizard")
                result = build_result
        else:
            result = subprocess.run(["npx", "-y", "@ember-ai/setup"], env=env)

        if result.returncode == 0:
            ctx.reload()

        return result.returncode
    except KeyboardInterrupt:
        print("\nSetup cancelled.")
        return 1
    except Exception as e:
        print(f"Error launching setup: {e}")
        return 1


def cmd_version(args: argparse.Namespace) -> int:
    try:
        import ember

        print(f"Ember {ember.__version__}")
    except AttributeError:
        print("Ember (version unknown)")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from ember.models.catalog import get_model_info, get_providers, list_available_models

    if args.providers:
        print("Available providers:")
        for provider in sorted(get_providers(include_dynamic=False)):
            print(f"  - {provider}")
    else:
        provider_filter = args.provider
        models = list_available_models(provider=provider_filter, include_dynamic=False)
        if provider_filter:
            print(f"Available {provider_filter} models:")
        else:
            print("Available models:")

        for model_id in models:
            description = ""
            try:
                info = get_model_info(model_id, include_dynamic=False)
            except KeyError:
                pass
            else:
                description = info.description or ""
            print(f"  {model_id:<20} {description}")

    return 0


def cmd_test(args: argparse.Namespace) -> int:
    ctx = args.context

    model = args.model or ctx.get_config("models.default", "gpt-3.5-turbo")

    try:
        print(f"Testing connection with {model}...")
        ctx.model_registry.get_model(model)

        response = ctx.model_registry.invoke_model(model, "Say hello!")
        print(f"Success: {response.data}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main CLI entry point.

    Initializes the Ember context, parses command line arguments, and
    dispatches to appropriate command handlers. Handles all exceptions
    and converts them to appropriate exit codes.

    Returns:
        int: Exit code for the shell
            - 0: Success
            - 1: General error
            - 2: Incorrect usage (shows help)
            - 130: Interrupted by user (Ctrl+C)

    Architecture Notes:
        - Context is initialized early and shared with all commands
        - Commands are organized as subcommands with dedicated parsers
        - All exceptions are caught and converted to exit codes
        - SystemExit is handled specially to preserve its exit code
    """
    # Initialize context early for all commands
    ctx = EmberContext.current()

    parser = argparse.ArgumentParser(
        prog="ember", description="Ember - Build Compound AI Systems with elegance"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Run interactive setup wizard")
    setup_parser.set_defaults(func=cmd_setup)

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version")
    version_parser.set_defaults(func=cmd_version)

    # Models command
    models_parser = subparsers.add_parser("models", help="List available models")
    models_parser.add_argument("--provider", help="Filter by provider")
    models_parser.add_argument("--providers", action="store_true", help="List providers instead")
    models_parser.set_defaults(func=cmd_models)

    # Test command
    test_parser = subparsers.add_parser("test", help="Test API connection")
    test_parser.add_argument("--model", help="Model to test with")
    test_parser.set_defaults(func=cmd_test)

    # Configure command
    config_parser = subparsers.add_parser("configure", help="Manage configuration")
    config_subparsers = config_parser.add_subparsers(dest="action", help="Action to perform")

    # configure get
    get_parser = config_subparsers.add_parser("get", help="Get configuration value")
    get_parser.add_argument("key", help="Configuration key (dot notation)")
    get_parser.add_argument("--default", help="Default value if not found")

    # configure set
    set_parser = config_subparsers.add_parser("set", help="Set configuration value")
    set_parser.add_argument("key", help="Configuration key (dot notation)")
    set_parser.add_argument("value", help="Value to set (JSON or string)")

    # configure list
    list_parser = config_subparsers.add_parser("list", help="List all configuration")
    list_parser.add_argument(
        "--format", choices=["yaml", "json"], default="yaml", help="Output format"
    )

    # configure show
    show_parser = config_subparsers.add_parser("show", help="Show configuration section")
    show_parser.add_argument("section", nargs="?", help="Section to show")
    show_parser.add_argument(
        "--format", choices=["yaml", "json"], default="yaml", help="Output format"
    )

    # configure migrate
    config_subparsers.add_parser("migrate", help="Migrate old configuration files")

    # configure import
    import_parser = config_subparsers.add_parser(
        "import", help="Import configuration from external tools"
    )
    import_parser.add_argument(
        "--config-path", type=Path, help="Path to external config file to import"
    )
    import_parser.add_argument(
        "--output-path",
        type=Path,
        help="Output path for Ember config (default: ~/.ember/config.yaml)",
    )
    import_parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        default=True,
        help="Skip backup of existing config before importing",
    )
    import_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without making changes",
    )

    config_parser.set_defaults(func=cmd_configure)

    # Context command
    context_parser = subparsers.add_parser("context", help="Inspect current context")
    context_subparsers = context_parser.add_subparsers(dest="action", help="Context actions")

    # context view
    view_parser = context_subparsers.add_parser("view", help="View current configuration")
    view_parser.add_argument(
        "--format", choices=["yaml", "json"], default="yaml", help="Output format"
    )
    view_parser.add_argument("--filter", help="Filter to specific path (dot notation)")
    view_parser.set_defaults(func=cmd_context_view)

    # context validate
    validate_parser = context_subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.set_defaults(func=cmd_context_validate)

    # Registry command
    registry_parser = subparsers.add_parser("registry", help="Inspect model and data registries")
    registry_subparsers = registry_parser.add_subparsers(dest="action", help="Registry actions")

    # registry list-models
    list_models_parser = registry_subparsers.add_parser("list-models", help="List available models")
    list_models_parser.add_argument("--provider", help="Filter by provider")
    list_models_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed information"
    )
    list_models_parser.set_defaults(func=cmd_registry_list_models)

    # registry list-providers
    list_providers_parser = registry_subparsers.add_parser(
        "list-providers", help="List configured providers"
    )
    list_providers_parser.set_defaults(func=cmd_registry_list_providers)

    # registry info
    info_parser = registry_subparsers.add_parser("info", help="Show model details")
    info_parser.add_argument("model_id", help="Model identifier")
    info_parser.set_defaults(func=cmd_registry_info)

    # Set default to help
    parser.set_defaults(func=None)

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command specified
    if args.func is None:
        parser.print_help()
        return 2  # Standard exit code for incorrect usage

    # Pass context to commands that need it
    args.context = ctx

    # Run command and return its exit code
    try:
        ret = args.func(args)
        return int(ret)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return 130  # Standard exit code for SIGINT
    except SystemExit as e:
        # Pass through SystemExit with its code
        return int(e.code) if isinstance(e.code, int) else 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
