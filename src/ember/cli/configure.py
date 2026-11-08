"""``ember configure``: show and change the config file.

Every change is checked against the rules the library applies when it loads
the file, and is written only if the result would load. Keys are read with
`getpass`, never from the command line, and are printed only masked.
"""

import argparse
import dataclasses
import getpass
import os
from types import MappingProxyType

import yaml

from ember import config, context
from ember.errors import ConfigError

__all__ = ["SETTABLE_FIELDS", "set_field", "set_key", "settable_field", "show", "unset"]

SETTABLE_FIELDS = ("kind", "base_url", "api_key_env")


def settable_field(field: str) -> str:
    """Parses the ``field`` argument of ``configure set``; ``api_key`` is refused."""
    if field == "api_key":
        raise argparse.ArgumentTypeError(
            "keys are never read from the command line; use: ember configure set-key PROVIDER"
        )
    if field not in SETTABLE_FIELDS:
        raise argparse.ArgumentTypeError(f"choose from {', '.join(SETTABLE_FIELDS)}")
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
    providers = context.from_config(config.load()).providers
    if provider not in providers:
        raise ConfigError(
            f"unknown provider {provider!r}; known: {', '.join(sorted(providers))}. "
            f"To add one: ember configure set {provider} kind openai_compatible"
        )
    settings = providers[provider]
    key = getpass.getpass(f"API key for {provider} (input hidden): ").strip()
    if not key:
        raise ConfigError("no key entered; nothing was saved")
    _save(provider, "api_key", key)
    print(f"saved the {provider} key to {config.config_path()}")
    if settings.api_key_env and os.environ.get(settings.api_key_env):
        print(f"note: {settings.api_key_env} is set and takes precedence over the saved key")
    return 0


def set_field(args: argparse.Namespace) -> int:
    """Sets a provider's kind, base_url or api_key_env."""
    provider: str = args.provider
    field: str = args.field
    _save(provider, field, args.value)
    print(f"set providers.{provider}.{field} in {config.config_path()}")
    return 0


def unset(args: argparse.Namespace) -> int:
    """Removes a provider field, or a provider, then checks that the file loads."""
    provider: str = args.provider
    field: str | None = args.field
    config.remove_provider_field(provider, field)
    target = f"providers.{provider}" if field is None else f"providers.{provider}.{field}"
    print(f"removed {target} from {config.config_path()}")
    try:
        context.from_config(config.load())
    except ConfigError as exc:
        raise ConfigError(f"the config still does not load: {exc}") from exc
    return 0


def _save(provider: str, field: str, value: str) -> None:
    """Writes one provider field if the resulting config would load."""
    current = config.load()
    fields = {**current.providers.get(provider, {}), field: value}
    providers = {**current.providers, provider: MappingProxyType(fields)}
    candidate = dataclasses.replace(current, providers=MappingProxyType(providers))
    try:
        context.from_config(candidate)
    except ConfigError as exc:
        raise ConfigError(f"nothing was saved: {exc}") from exc
    config.save_provider_field(provider, field, value)
