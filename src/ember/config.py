"""The ember config file.

This module is the only code that reads or writes the config file. The file
lives at ``$EMBER_CONFIG`` or ``~/.ember/config.yaml``. A missing file means
defaults; an unreadable or invalid file raises `ConfigError` naming the path.

Example file::

    default_model: openai/gpt-5
    providers:
      openai:
        api_key: sk-...
      local:
        kind: openai_compatible
        base_url: http://localhost:11434/v1
    max_in_flight_per_provider: 64
    retry:
      max_attempts: 3
"""

import dataclasses
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from ember.errors import ConfigError

__all__ = ["ConfigData", "config_path", "load", "redacted", "save_provider_field"]

_TOP_LEVEL_KEYS = frozenset(
    {"default_model", "providers", "require_price", "max_in_flight_per_provider", "retry"}
)
_PROVIDER_KEYS = frozenset({"kind", "api_key", "api_key_env", "base_url"})
_RETRY_KEYS = frozenset({"max_attempts", "base_delay_s", "max_delay_s"})
_SECRET_KEYS = frozenset({"api_key"})


@dataclasses.dataclass(frozen=True, slots=True)
class ConfigData:
    """The validated contents of the config file.

    Attributes:
        providers: Raw per-provider fields by provider name.
        default_model: Model used when a call names none.
        require_price: Refuse requests whose price is unknown.
        max_in_flight_per_provider: Concurrent requests allowed per provider.
        retry: Raw retry policy fields.
    """

    providers: Mapping[str, Mapping[str, str]] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    default_model: str | None = None
    require_price: bool = False
    max_in_flight_per_provider: int = 64
    retry: Mapping[str, float] = dataclasses.field(default_factory=lambda: MappingProxyType({}))


def config_path() -> Path:
    """Returns the config file location: ``$EMBER_CONFIG`` or ``~/.ember/config.yaml``."""
    override = os.environ.get("EMBER_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ember" / "config.yaml"


def load(path: Path | None = None) -> ConfigData:
    """Reads and validates the config file.

    Args:
        path: File to read; defaults to `config_path()`.

    Returns:
        The validated configuration, or defaults when the file does not exist.

    Raises:
        ConfigError: The file cannot be read, is not valid YAML, or contains
            unknown keys or values of the wrong type.
    """
    path = path or config_path()
    raw = _read_yaml(path)
    return _validate(raw, path)


def save_provider_field(provider: str, field: str, value: str, path: Path | None = None) -> None:
    """Sets one provider field in the config file, atomically, with mode 0600.

    Args:
        provider: Provider name.
        field: One of ``kind``, ``api_key``, ``api_key_env``, ``base_url``.
        value: The new value.
        path: File to write; defaults to `config_path()`.

    Raises:
        ConfigError: The field is unknown or the file cannot be written.
    """
    if field not in _PROVIDER_KEYS:
        raise ConfigError(
            f"unknown provider field {field!r}; expected one of {sorted(_PROVIDER_KEYS)}"
        )
    path = path or config_path()
    raw = _read_yaml(path)
    providers = raw.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise ConfigError(f"{path}: 'providers' must be a mapping")
    entry = providers.setdefault(provider, {})
    if not isinstance(entry, dict):
        raise ConfigError(f"{path}: providers.{provider} must be a mapping")
    entry[field] = value
    _validate(raw, path)
    _write_atomic(path, yaml.safe_dump(raw, sort_keys=True))


def redacted(data: ConfigData) -> dict[str, Any]:
    """Returns the configuration as plain data with every credential masked."""
    providers = {
        name: {key: _mask(value) if key in _SECRET_KEYS else value for key, value in fields.items()}
        for name, fields in data.providers.items()
    }
    return {
        "default_model": data.default_model,
        "providers": providers,
        "require_price": data.require_price,
        "max_in_flight_per_provider": data.max_in_flight_per_provider,
        "retry": dict(data.retry),
    }


def _mask(secret: str) -> str:
    return "…" if len(secret) < 12 else f"…{secret[-4:]}"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc.strerror}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" line {mark.line + 1}" if mark is not None else ""
        raise ConfigError(f"{path}{where}: not valid YAML") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: the top level must be a mapping")
    return raw


def _validate(raw: Mapping[str, Any], path: Path) -> ConfigData:
    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    providers = _validate_providers(raw.get("providers", {}), path)
    retry = raw.get("retry", {})
    if not isinstance(retry, dict) or set(retry) - _RETRY_KEYS:
        raise ConfigError(f"{path}: retry accepts only {sorted(_RETRY_KEYS)}")
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in retry.values()):
        raise ConfigError(f"{path}: retry values must be numbers")
    default_model = raw.get("default_model")
    if default_model is not None and not isinstance(default_model, str):
        raise ConfigError(f"{path}: default_model must be a string")
    require_price = raw.get("require_price", False)
    if not isinstance(require_price, bool):
        raise ConfigError(f"{path}: require_price must be true or false")
    in_flight = raw.get("max_in_flight_per_provider", 64)
    if not isinstance(in_flight, int) or isinstance(in_flight, bool):
        raise ConfigError(f"{path}: max_in_flight_per_provider must be an integer")
    return ConfigData(
        providers=MappingProxyType(providers),
        default_model=default_model,
        require_price=require_price,
        max_in_flight_per_provider=in_flight,
        retry=MappingProxyType(dict(retry)),
    )


def _validate_providers(raw: object, path: Path) -> dict[str, Mapping[str, str]]:
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: providers must be a mapping")
    providers: dict[str, Mapping[str, str]] = {}
    for name, fields in raw.items():
        if not isinstance(name, str) or not name or "/" in name:
            raise ConfigError(f"{path}: provider names must be non-empty and contain no '/'")
        if not isinstance(fields, dict):
            raise ConfigError(f"{path}: providers.{name} must be a mapping")
        unknown = set(fields) - _PROVIDER_KEYS
        if unknown:
            raise ConfigError(f"{path}: providers.{name}: unknown keys {sorted(unknown)}")
        if not all(isinstance(v, str) for v in fields.values()):
            raise ConfigError(f"{path}: providers.{name}: values must be strings")
        providers[name] = MappingProxyType(dict(fields))
    return providers


def _write_atomic(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config-", suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise ConfigError(f"cannot write {path}: {exc.strerror}") from exc
