"""`ember configure` command."""

import argparse
import json
import sys

import yaml


def cmd_configure(args: argparse.Namespace) -> int:
    ctx = args.context

    if args.action is None:
        from ember.cli.main import cmd_setup

        return cmd_setup(args)

    if args.action == "get":
        key = args.key
        if not isinstance(key, str) or not key:
            print("Error: configuration key is required", file=sys.stderr)
            return 1

        value = ctx.get_config(key, args.default)
        if value is None and args.default is None:
            print(f"Key '{key}' not found", file=sys.stderr)
            return 1
        print(value)
        return 0

    if args.action == "set":
        key = args.key
        if not isinstance(key, str) or not key:
            print("Error: configuration key is required", file=sys.stderr)
            return 1

        if args.value is None:
            print("Error: value is required for set operation", file=sys.stderr)
            return 1

        try:
            value = json.loads(args.value)
        except json.JSONDecodeError:
            value = args.value

        is_sensitive = any(
            pattern in key.lower() for pattern in ("api_key", "secret", "password", "token")
        )

        if is_sensitive:
            print(f"Warning: setting sensitive key '{key}'", file=sys.stderr)

        try:
            ctx.set_config(key, value)
            ctx.save()
            print(f"Updated {key}")
            if not is_sensitive:
                print(f"Value: {value}")
        except (ValueError, TypeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0

    if args.action == "list":
        config = ctx.get_all_config()
        if args.format == "json":
            print(json.dumps(config, indent=2))
        else:
            print(yaml.dump(config, default_flow_style=False))
        return 0

    if args.action == "show":
        if args.section:
            config = ctx.get_config(args.section, {})
        else:
            config = ctx.get_all_config()

        if args.format == "json":
            print(json.dumps(config, indent=2))
        else:
            print(yaml.dump(config, default_flow_style=False))
        return 0

    if args.action == "migrate":
        from ember._internal.configuration.migrations import main as migrate_main

        migrate_main()
        return 0

    if args.action == "import":
        from ember.cli.commands.config_import import cmd_import

        return cmd_import(args)

    print(f"Unknown action: {args.action}", file=sys.stderr)
    return 2
