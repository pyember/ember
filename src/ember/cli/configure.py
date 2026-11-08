"""``ember configure``: show and change the config file.

Output never repeats a name or value typed on the command line unless it is
already known (a provider in the file or built in, a field name, a kind),
because a key typed in the wrong place must never be printed.

Every change is checked against the rules the library applies when it loads
the file, and is written only if the result would load; a file that already
fails to load is changed anyway, since the change may be its repair. Keys are
read with `getpass`, never from the command line, and are printed only masked.
A config file with comments is never rewritten, because rewriting would drop
them: edit it by hand, or remove its comments first.
"""

import argparse
import dataclasses
import getpass
import os
import typing
from collections.abc import Callable, Mapping
from types import MappingProxyType

import yaml

from ember import config, context
from ember.errors import ConfigError

__all__ = [
    "FIELD_CHOICES",
    "KEYS_NOT_IN_ARGV",
    "SETTABLE_FIELDS",
    "set_field",
    "set_key",
    "settable_field",
    "show",
    "unset",
]

SETTABLE_FIELDS = ("kind", "base_url", "api_key_env")
KINDS: tuple[str, ...] = typing.get_args(context.ProviderKind)
KEYS_NOT_IN_ARGV = (
    "keys are never read from the command line; use: ember configure set-key PROVIDER"
)
FIELD_CHOICES = f"FIELD must be one of {', '.join(SETTABLE_FIELDS)}"


def settable_field(field: str) -> str:
    """Parses the ``field`` argument of ``configure set``; ``api_key`` is refused."""
    if field == "api_key":
        raise argparse.ArgumentTypeError(KEYS_NOT_IN_ARGV)
    if field not in SETTABLE_FIELDS:
        raise argparse.ArgumentTypeError(FIELD_CHOICES)
    return field


def show(args: argparse.Namespace) -> int:
    """Prints the config file with every key masked, then checks that it loads."""
    del args
    path = config.config_path()
    data = config.load(path)
    state = "" if path.exists() else " (does not exist; these are the defaults)"
    print(f"# {path}{state}")
    print(yaml.safe_dump(config.redacted(data), sort_keys=False, allow_unicode=True), end="")
    context.from_config(data)
    return 0


def set_key(args: argparse.Namespace) -> int:
    """Reads a key at a hidden prompt and saves it for a known provider."""
    provider: str = args.provider
    known = _known_providers()
    if provider not in known:
        raise _unknown_provider(known)
    providers = context.from_config(config.load()).providers
    try:
        key = getpass.getpass(f"API key for {provider} (input hidden): ").strip()
    except EOFError:
        key = ""
    if not key:
        raise ConfigError("no key entered; nothing was saved")
    _change(
        lambda data: _with_field(data, provider, "api_key", key),
        lambda: config.save_provider_field(provider, "api_key", key),
        done=f"saved the {provider} key to {config.config_path()}",
    )
    env = providers[provider].api_key_env
    if env and os.environ.get(env):
        print(f"note: {env} is set and takes precedence over the saved key")
    return 0


def set_field(args: argparse.Namespace) -> int:
    """Sets a provider's kind, base_url or api_key_env."""
    provider: str = args.provider
    field: str = args.field
    value: str = args.value
    if field == "kind" and value not in KINDS:
        raise ConfigError(f"KIND must be one of {', '.join(KINDS)}")
    known = _known_providers()
    new = provider not in known
    if new and field != "kind":
        raise _unknown_provider(known)
    path = config.config_path()
    _change(
        lambda data: _with_field(data, provider, field, value),
        lambda: config.save_provider_field(provider, field, value),
        done=(
            f"added the new provider, kind {value}, to {path}"
            if new
            else f"set providers.{provider}.{field} in {path}"
        ),
    )
    return 0


def unset(args: argparse.Namespace) -> int:
    """Removes a provider field, or a whole provider."""
    provider: str = args.provider
    field: str | None = args.field
    path = config.config_path()
    entries = config.provider_fields(path)
    if provider not in entries:
        there = ", ".join(sorted(entries)) or "none"
        raise ConfigError(f"PROVIDER is not in {path}; providers there: {there}")
    if field is not None and field not in entries[provider]:
        there = ", ".join(entries[provider]) or "none"
        raise ConfigError(f"FIELD is not set for {provider}; fields set there: {there}")
    target = f"providers.{provider}" if field is None else f"providers.{provider}.{field}"
    _change(
        lambda data: _without_field(data, provider, field),
        lambda: config.remove_provider_field(provider, field),
        done=f"removed {target} from {config.config_path()}",
    )
    return 0


def _change(
    changed: Callable[[config.ConfigData], config.ConfigData],
    write: Callable[[], None],
    *,
    done: str,
) -> None:
    """Writes a change only if the config still loads after it.

    When the file already fails to load, the change is written anyway, since it
    may be the repair, and the command fails if the file still does not load.
    """
    try:
        current = config.load()
        context.from_config(current)
    except ConfigError:
        write()
        print(done)
        try:
            context.from_config(config.load())
        except ConfigError as exc:
            raise ConfigError(f"the config still does not load: {exc}") from None
        return
    try:
        context.from_config(changed(current))
    except ConfigError as exc:
        raise ConfigError(f"nothing was saved: {exc}") from None
    write()
    print(done)


def _known_providers() -> Mapping[str, tuple[str, ...]]:
    """Providers that exist before the command runs: the built-in ones and the file's."""
    builtin: dict[str, tuple[str, ...]] = {name: () for name in context.BUILTIN_PROVIDERS}
    return {**builtin, **config.provider_fields()}


def _unknown_provider(known: Mapping[str, tuple[str, ...]]) -> ConfigError:
    # The name is not repeated: it came from the command line and may be a key.
    return ConfigError(
        f"PROVIDER is not a configured provider; known: {', '.join(sorted(known))}. "
        "To add one: ember configure set NAME kind openai_compatible"
    )


def _with_field(
    data: config.ConfigData, provider: str, field: str, value: str
) -> config.ConfigData:
    fields = {**data.providers.get(provider, {}), field: value}
    providers = {**data.providers, provider: MappingProxyType(fields)}
    return dataclasses.replace(data, providers=MappingProxyType(providers))


def _without_field(data: config.ConfigData, provider: str, field: str | None) -> config.ConfigData:
    providers = {name: fields for name, fields in data.providers.items() if name != provider}
    if field is not None:
        rest = {k: v for k, v in data.providers.get(provider, {}).items() if k != field}
        if rest:
            providers[provider] = MappingProxyType(rest)
    return dataclasses.replace(data, providers=MappingProxyType(providers))
