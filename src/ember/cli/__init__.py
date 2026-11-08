"""The ``ember`` command.

Arguments are parsed before anything else happens: no config file is read and
nothing is constructed until a command runs, so a broken config file never
stops ``ember configure`` from repairing it. Errors ember raises on purpose
print as one line and exit with status 1; usage errors exit with status 2.
"""

import argparse
import sys
from collections.abc import Callable, Sequence

import ember
from ember.cli import configure
from ember.errors import ConfigError, ModelError

__all__ = ["main"]

Handler = Callable[[argparse.Namespace], int]


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the ``ember`` command and returns its exit status.

    Args:
        argv: Arguments after the program name; defaults to ``sys.argv[1:]``.
    """
    parser = _parser()
    args, extras = parser.parse_known_args(argv)
    if extras:
        # argparse would quote them, and a misplaced key must never be echoed.
        parser.error("unexpected extra arguments (not shown)")
    handler: Handler = args.handler
    try:
        return handler(args)
    except (ConfigError, ModelError) as exc:
        print(f"ember: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ember", description="Compose and evaluate model calls.")
    parser.add_argument("--version", action="version", version=f"ember {ember.__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    configure_parser = commands.add_parser(
        "configure", help="show or change the config file", description=configure.__doc__
    )
    actions = configure_parser.add_subparsers(dest="action", required=True, metavar="ACTION")
    actions.add_parser("show", help="print the config file with keys masked").set_defaults(
        handler=configure.show
    )
    set_key = actions.add_parser("set-key", help="save a provider's key, typed at a hidden prompt")
    set_key.add_argument("provider")
    set_key.set_defaults(handler=configure.set_key)
    set_field = actions.add_parser("set", help="set a provider's kind, base_url or api_key_env")
    set_field.add_argument("provider")
    set_field.add_argument(
        "field",
        type=configure.settable_field,
        metavar="{" + ",".join(configure.SETTABLE_FIELDS) + "}",
    )
    set_field.add_argument("value")
    set_field.set_defaults(handler=configure.set_field)
    unset = actions.add_parser("unset", help="remove a provider field, or a whole provider")
    unset.add_argument("provider")
    unset.add_argument("field", nargs="?", help="any field name, including a misspelt one")
    unset.set_defaults(handler=configure.unset)
    return parser
