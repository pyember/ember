"""Configuration file loader.

Ember treats `~/.ember/config.yaml` as the single source of truth for runtime
configuration. Configuration loading is therefore deterministic: we do not
perform environment-variable substitution inside configuration files.
"""

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

import yaml

Config: TypeAlias = dict[str, object]


def load_config(path: str | Path) -> Config:
    """Load configuration from file.

    Args:
        path: Path to configuration file.

    Returns:
        dict[str, object]: Configuration dictionary. Empty dict if file contains no data.

    Raises:
        FileNotFoundError: If specified file doesn't exist.
        ValueError: If file format is not supported or content is invalid.
            Includes detailed parse error for debugging.
        TypeError: If the parsed configuration is not a mapping with string keys.

    Format Detection:
        1. Uses file extension if present (.json, .yaml, .yml)
        2. For extensionless files, examines content:
           - Files starting with '{' are treated as JSON
           - All others are treated as YAML
        3. If detection fails, tries alternate format before failing

    Examples:
        >>> config = load_config("~/.ember/config.yaml")  # doctest: +SKIP
        >>> config = load_config(Path.home() / ".ember" / "config")  # doctest: +SKIP
    """
    config_path = Path(path).expanduser()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Read file content
    content = config_path.read_text(encoding="utf-8")

    # Auto-detect format if no extension
    suffix = config_path.suffix.lower()
    if not suffix:
        stripped = content.lstrip()
        suffix = ".json" if stripped.startswith("{") else ".yaml"

    # Parse based on format and fail fast on invalid payloads
    try:
        if suffix in (".yaml", ".yml"):
            parsed = yaml.safe_load(content)
            config = {} if parsed is None else parsed
        elif suffix == ".json":
            config = json.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse config file: {config_path}") from exc

    if not isinstance(config, Mapping):
        raise TypeError(
            f"Config file {config_path} must contain a mapping, got {type(config).__name__}"
        )

    normalized: dict[str, object] = {}
    for key, value in config.items():
        if not isinstance(key, str):
            raise TypeError(
                f"Config file {config_path} contains a non-string key: "
                f"{key!r} ({type(key).__name__})"
            )
        normalized[key] = value

    return normalized


def save_config(config: Mapping[str, object], path: str | Path) -> None:
    """Save configuration to file.

    Saves configuration in YAML or JSON format based on file extension.
    Creates parent directories if needed. Uses atomic writes for safety.

    Args:
        config: Configuration mapping to save.
        path: Destination file path. Format determined by extension:
            - .yaml/.yml -> YAML format
            - .json or no extension -> JSON format

    File Format:
        - YAML: Human-readable, preserves key order, no flow style
        - JSON: Pretty-printed with 2-space indentation

    Safety:
        - Creates parent directories with parents=True
        - Writes atomically via a temporary file + replace
        - Sets file permissions to 0600

    Examples:
        >>> save_config({"models": {"default": "gpt-4"}}, "config.yaml")
        >>> save_config(config_dict, Path.home() / ".ember" / "config")
    """
    config_path = Path(path).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = config_path.suffix.lower()
    if not suffix:
        suffix = ".json"

    if suffix in (".yaml", ".yml"):
        rendered = yaml.safe_dump(config, default_flow_style=False, sort_keys=False)
    else:
        rendered = json.dumps(config, indent=2)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(config_path.parent),
        prefix=f".{config_path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, config_path)
    finally:
        tmp_path.unlink(missing_ok=True)
