"""The ``ember`` command.

Arguments are parsed before anything else happens: no config file is read and
nothing is constructed until a command runs, so a broken config file never
stops ``ember configure`` from repairing it. Errors ember raises on purpose
print as one line and exit with status 1; usage errors exit with status 2.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import NoReturn

import ember
from ember.cli import configure
from ember.errors import ConfigError, ModelError

__all__ = ["main"]

Handler = Callable[[argparse.Namespace], int]

EXTRA_ARGUMENTS = "unexpected extra arguments (not shown)"
# Messages this package writes itself; any other argparse message is replaced.
_OWN_MESSAGES = (EXTRA_ARGUMENTS, configure.KEYS_NOT_IN_ARGV, configure.FIELD_CHOICES)


class _Parser(argparse.ArgumentParser):
    """An argument parser whose errors never quote what was typed.

    Some argparse errors quote the offending argument ("invalid choice:
    'sk-...'"), and a key typed in the wrong place must never be printed. Each
    message is therefore one of this package's own, or is built only from the
    parser's definition: its program name, usage and subcommand names.

    Attributes:
        commands: The subcommands this parser accepts, for its error message.
    """

    commands: tuple[str, ...] = ()

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {self._safe_message(message)}\n")

    def _safe_message(self, message: str) -> str:
        for own in _OWN_MESSAGES:
            if message.endswith(own):
                return own
        if "invalid choice" in message and self.commands:
            return f"unknown command; choose from {', '.join(self.commands)}"
        if "required" in message:
            return "missing arguments; see the usage above"
        return "invalid arguments; see the usage above"


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the ``ember`` command and returns its exit status.

    Args:
        argv: Arguments after the program name; defaults to ``sys.argv[1:]``.
    """
    parser = _parser()
    args, extras = parser.parse_known_args(argv)
    if extras:
        # argparse would quote them, and a misplaced key must never be echoed.
        parser.error(EXTRA_ARGUMENTS)
    handler: Handler = args.handler
    try:
        return handler(args)
    except (ConfigError, ModelError) as exc:
        print(f"ember: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


def _parser() -> _Parser:
    parser = _Parser(prog="ember", description="Compose and evaluate model calls.")
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
    parser.commands = tuple(commands.choices)
    configure_parser.commands = tuple(actions.choices)
    return parser
