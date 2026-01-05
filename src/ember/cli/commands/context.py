"""`ember context` subcommands."""

import argparse
import json

import yaml

from ember.context import context
from ember.core.credentials import CredentialNotFoundError


def cmd_context_view(args: argparse.Namespace) -> int:
    ctx = context.get()
    config = ctx.get_all_config()

    filter_path = args.filter
    if filter_path:
        filtered: object = config
        for part in filter_path.split("."):
            if not isinstance(filtered, dict) or part not in filtered:
                print(f"Path '{filter_path}' not found in configuration")
                return 1
            filtered = filtered[part]
        config = {filter_path: filtered}

    if args.format == "json":
        print(json.dumps(config, indent=2, sort_keys=True))
    else:
        print(yaml.dump(config, default_flow_style=False, sort_keys=True))
    return 0


def cmd_context_validate(args: argparse.Namespace) -> int:
    del args

    ctx = context.get()
    issues: list[str] = []

    for provider in ("openai", "anthropic", "google"):
        try:
            ctx.get_credential(provider)
        except CredentialNotFoundError:
            if ctx.get_config(f"providers.{provider}"):
                issues.append(f"Missing API key for {provider}")

    default_model = ctx.get_config("models.default")
    if isinstance(default_model, str) and default_model:
        available_models = ctx.list_models()
        if default_model not in available_models:
            issues.append(f"Default model '{default_model}' is not available")

    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("Configuration is valid")
    return 0
