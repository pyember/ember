"""The ember config file.

This module is the only code that reads or writes the config file. The file
lives at ``$EMBER_CONFIG`` or ``~/.ember/config.yaml``. A missing file means
defaults; an unreadable or invalid file raises `ConfigError` naming the path.

The writers keep the file's key order, refuse to rewrite a file that has
comments (they would be lost), and replace the file atomically. They do not
lock it: two processes writing at the same moment can lose one of the updates.

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
import math
import os
import re
import sys
import tempfile
import urllib.parse
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from ember.errors import ConfigError

__all__ = [
    "ConfigData",
    "config_path",
    "load",
    "provider_fields",
    "redacted",
    "remove_provider_field",
    "save_provider_field",
]

_TOP_LEVEL_KEYS = frozenset(
    {"default_model", "providers", "require_price", "max_in_flight_per_provider", "retry"}
)
_PROVIDER_KEYS = frozenset({"kind", "api_key", "api_key_env", "base_url"})
_RETRY_KEYS = frozenset({"max_attempts", "base_delay_s", "max_delay_s"})
_SECRET_KEYS = frozenset({"api_key"})
_URL_KEYS = frozenset({"base_url"})
# Names and non-secret values are shown unmasked, so each has a format a
# misplaced key cannot pass; error messages describe the format and never
# repeat the value.
_PROVIDER_NAME = re.compile(r"[a-z][a-z0-9_]{0,31}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclasses.dataclass(frozen=True, slots=True)
class ConfigData:
    """The validated contents of the config file.

    Attributes:
        providers: Raw per-provider fields by provider name.
        default_model: Model used when a call names none.
        require_price: Refuse requests whose price is unknown.
        max_in_flight_per_provider: Concurrent requests allowed per provider.
        retry: Raw retry policy fields.
        path: The file this was read from, which errors about it name; ``None``
            for data that did not come from a file.
    """

    providers: Mapping[str, Mapping[str, str]] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    default_model: str | None = None
    require_price: bool = False
    max_in_flight_per_provider: int = 64
    retry: Mapping[str, float] = dataclasses.field(default_factory=lambda: MappingProxyType({}))
    path: Path | None = None


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
            unknown keys or values of the wrong type. The message names the
            file and, when the problem is at a key, its line.
    """
    return _validate(_read(path or config_path()))


def save_provider_field(provider: str, field: str, value: str, path: Path | None = None) -> None:
    """Sets one provider field in the config file, atomically, with mode 0600.

    Args:
        provider: Provider name.
        field: One of ``kind``, ``api_key``, ``api_key_env``, ``base_url``.
        value: The new value.
        path: File to write; defaults to `config_path()`.

    Raises:
        ConfigError: The field is unknown, the file has comments, the resulting
            file would be invalid, or the file cannot be written.
    """
    if field not in _PROVIDER_KEYS:
        raise ConfigError(
            f"unknown provider field {field!r}; expected one of {sorted(_PROVIDER_KEYS)}"
        )
    doc = _read_for_rewrite(path or config_path())
    providers = doc.raw.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise doc.error("providers must be a mapping", "providers")
    entry = providers.setdefault(provider, {})
    if not isinstance(entry, dict):
        raise doc.error(f"providers.{provider} must be a mapping", "providers", provider)
    entry[field] = value
    _rewrite(doc)


def remove_provider_field(provider: str, field: str | None, path: Path | None = None) -> None:
    """Removes one provider field, or the whole provider entry, atomically.

    ``field`` may be any name, not only a known one, so a misspelt key that
    makes the file invalid can be removed. The result is validated before it
    is written.

    Args:
        provider: Provider name.
        field: Field to remove, or ``None`` to remove the provider entry.
        path: File to write; defaults to `config_path()`.

    Raises:
        ConfigError: Nothing by that name is set, the file cannot be read or
            has comments, the resulting file would be invalid, or it cannot be
            written.
    """
    doc = _read_for_rewrite(path or config_path())
    target = f"providers.{provider}" if field is None else f"providers.{provider}.{field}"
    providers = doc.raw.get("providers")
    if not isinstance(providers, dict) or provider not in providers:
        raise ConfigError(f"{doc.path}: {target} is not set")
    if field is None:
        del providers[provider]
    else:
        entry = providers[provider]
        if not isinstance(entry, dict) or field not in entry:
            raise ConfigError(f"{doc.path}: {target} is not set")
        del entry[field]
        if not entry:
            del providers[provider]
    if not providers:
        del doc.raw["providers"]
    _rewrite(doc)


def provider_fields(path: Path | None = None) -> Mapping[str, tuple[str, ...]]:
    """Returns each provider entry in the file and the names of the fields it sets.

    Values are not validated, so this also reads a file that `load` rejects;
    `ember configure` uses it to check what a repair would remove.

    Raises:
        ConfigError: The file cannot be read, is not valid YAML, or its
            ``providers`` is not a mapping.
    """
    doc = _read(path or config_path())
    providers = doc.raw.get("providers", {})
    if not isinstance(providers, dict):
        raise doc.error("providers must be a mapping", "providers")
    return MappingProxyType(
        {
            str(name): tuple(str(key) for key in fields) if isinstance(fields, dict) else ()
            for name, fields in providers.items()
        }
    )


def redacted(data: ConfigData) -> dict[str, Any]:
    """Returns the configuration as plain data with every credential masked.

    Keys show only their last four characters, and URLs lose their user
    information and query string, where credentials are sometimes embedded.
    """
    providers = {
        name: {key: _redact_field(key, value) for key, value in fields.items()}
        for name, fields in data.providers.items()
    }
    return {
        "default_model": data.default_model,
        "providers": providers,
        "require_price": data.require_price,
        "max_in_flight_per_provider": data.max_in_flight_per_provider,
        "retry": dict(data.retry),
    }


def _redact_field(key: str, value: str) -> str:
    if key in _SECRET_KEYS:
        return "…" if len(value) < 12 else f"…{value[-4:]}"
    if key in _URL_KEYS:
        return _redact_url(value)
    return value


def _redact_url(url: str) -> str:
    """Masks user information and the query string, with or without a scheme."""
    scheme, separator, rest = url.partition("://")
    prefix = f"{scheme}{separator}" if separator else ""
    if not separator:
        rest = url
    end = min([index for index in map(rest.find, "/?#") if index >= 0], default=len(rest))
    authority, tail = rest[:end], rest[end:]
    if "@" in authority:
        authority = "…@" + authority.rpartition("@")[2]
    path, question, query_and_fragment = tail.partition("?")
    if question:
        _, hash_mark, fragment = query_and_fragment.partition("#")
        tail = f"{path}?…{hash_mark}{fragment}"
    return f"{prefix}{authority}{tail}"


@dataclasses.dataclass(frozen=True, slots=True)
class _Document:
    """A parsed config file and the line of each mapping key, for error messages."""

    path: Path
    text: str
    raw: dict[str, Any]
    key_lines: Mapping[tuple[str, ...], int]

    def error(self, message: str, *keys: str) -> ConfigError:
        """Returns an error naming the file and, when known, the line of ``keys``."""
        line = self.key_lines.get(keys)
        where = str(self.path) if line is None else f"{self.path} line {line}"
        return ConfigError(f"{where}: {message}")


def _read(path: Path) -> _Document:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _Document(path, "", {}, {})
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc.strerror}") from exc
    try:
        raw = yaml.safe_load(text)
        node = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        # Not chained: PyYAML's message quotes the offending line, which may hold a key.
        raise _syntax_error(path, exc) from None
    if raw is None:
        return _Document(path, text, {}, {})
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: the top level must be a mapping")
    return _Document(path, text, raw, _key_lines(node, ()))


def _syntax_error(path: Path, exc: yaml.YAMLError) -> ConfigError:
    if isinstance(exc, yaml.MarkedYAMLError) and exc.problem_mark is not None:
        problem = f" ({exc.problem})" if exc.problem else ""
        return ConfigError(f"{path} line {exc.problem_mark.line + 1}: not valid YAML{problem}")
    return ConfigError(f"{path}: not valid YAML")


def _key_lines(node: yaml.Node | None, prefix: tuple[str, ...]) -> dict[tuple[str, ...], int]:
    lines: dict[tuple[str, ...], int] = {}
    if not isinstance(node, yaml.MappingNode):
        return lines
    for key_node, value_node in node.value:
        if isinstance(key_node, yaml.ScalarNode):
            keys = (*prefix, str(key_node.value))
            lines[keys] = key_node.start_mark.line + 1
            lines.update(_key_lines(value_node, keys))
    return lines


def _validate(doc: _Document) -> ConfigData:
    raw = doc.raw
    unknown = [str(key) for key in raw if key not in _TOP_LEVEL_KEYS]
    if unknown:
        raise doc.error(f"unknown keys {unknown}", unknown[0])
    providers = _validate_providers(doc)
    retry = _validate_retry(doc)
    default_model = raw.get("default_model")
    if default_model is not None and not isinstance(default_model, str):
        raise doc.error("default_model must be a string", "default_model")
    require_price = raw.get("require_price", False)
    if not isinstance(require_price, bool):
        raise doc.error("require_price must be true or false", "require_price")
    in_flight = raw.get("max_in_flight_per_provider", 64)
    if not isinstance(in_flight, int) or isinstance(in_flight, bool):
        raise doc.error(
            "max_in_flight_per_provider must be an integer", "max_in_flight_per_provider"
        )
    return ConfigData(
        providers=MappingProxyType(providers),
        default_model=default_model,
        require_price=require_price,
        max_in_flight_per_provider=in_flight,
        retry=MappingProxyType(retry),
        path=doc.path,
    )


def _validate_retry(doc: _Document) -> dict[str, float]:
    retry = doc.raw.get("retry", {})
    if not isinstance(retry, dict):
        raise doc.error("retry must be a mapping", "retry")
    unknown = [str(key) for key in retry if key not in _RETRY_KEYS]
    if unknown:
        raise doc.error(f"retry accepts only {sorted(_RETRY_KEYS)}", "retry", unknown[0])
    for key, value in retry.items():
        number = isinstance(value, int | float) and not isinstance(value, bool)
        if not number or (isinstance(value, float) and not math.isfinite(value)):
            raise doc.error(f"retry.{key} must be a finite number", "retry", key)
    return dict(retry)


def _validate_providers(doc: _Document) -> dict[str, Mapping[str, str]]:
    raw = doc.raw.get("providers", {})
    if not isinstance(raw, dict):
        raise doc.error("providers must be a mapping", "providers")
    providers: dict[str, Mapping[str, str]] = {}
    for name, fields in raw.items():
        if not isinstance(name, str) or not _PROVIDER_NAME.fullmatch(name):
            raise doc.error(
                "provider names must be lowercase letters, digits and _, starting with a letter, "
                "at most 32 characters",
                "providers",
                str(name),
            )
        if not isinstance(fields, dict):
            raise doc.error(f"providers.{name} must be a mapping", "providers", name)
        unknown = [str(key) for key in fields if key not in _PROVIDER_KEYS]
        if unknown:
            raise doc.error(
                f"providers.{name}: unknown keys {unknown}", "providers", name, unknown[0]
            )
        for key, value in fields.items():
            problem = _field_problem(key, value)
            if problem is not None:
                raise doc.error(f"providers.{name}.{key} {problem}", "providers", name, key)
        providers[name] = MappingProxyType(dict(fields))
    return providers


def _read_for_rewrite(path: Path) -> _Document:
    """Reads a file that is about to be rewritten; refuses one with comments."""
    doc = _read(path)
    comment_line = _first_comment_line(doc.text)
    if comment_line is not None:
        raise ConfigError(
            f"{path} line {comment_line}: the file has comments, which rewriting it would "
            "drop; edit it by hand, or remove its comments and run the command again"
        )
    return doc


def _rewrite(doc: _Document) -> None:
    """Validates the changed document and replaces the file with it."""
    _validate(doc)
    _write_atomic(doc.path, yaml.safe_dump(doc.raw, sort_keys=False, allow_unicode=True))


def _first_comment_line(text: str) -> int | None:
    scalars = [
        (token.start_mark.index, token.end_mark.index)
        for token in yaml.scan(text, Loader=yaml.SafeLoader)
        if isinstance(token, yaml.ScalarToken)
    ]
    for match in re.finditer("#", text):
        if not any(start <= match.start() < end for start, end in scalars):
            return text.count("\n", 0, match.start()) + 1
    return None


def _field_problem(key: str, value: object) -> str | None:
    """Says what is wrong with a provider field's value, without repeating it."""
    if not isinstance(value, str):
        return "must be a string"
    if key in _URL_KEYS and not _is_http_url(value):
        return "must be an http or https URL"
    if key == "api_key_env" and not _ENV_NAME.fullmatch(value):
        return "must be an environment variable name (letters, digits and _)"
    return None


def _is_http_url(value: str) -> bool:
    try:
        parts = urllib.parse.urlsplit(value)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.hostname)


def _write_atomic(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config-", suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        _sync_directory(path.parent)
    except OSError as exc:
        raise ConfigError(f"cannot write {path}: {exc.strerror}") from exc


def _sync_directory(directory: Path) -> None:
    """Makes the rename that replaced the file durable."""
    if sys.platform == "win32":
        return  # Windows cannot open a directory to fsync it; its renames need no such step.
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
