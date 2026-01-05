"""Credential storage utilities for Ember.

Credentials are stored in ~/.ember/config.yaml under the providers section:

    providers:
      openai:
        api_key: "sk-..."
      anthropic:
        api_key: "sk-ant-..."

This consolidates all Ember configuration into a single file.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class CredentialError(RuntimeError):
    """Base class for credential related failures."""


class CredentialNotFoundError(CredentialError):
    """Raised when a credential cannot be located."""


@dataclass(slots=True)
class CredentialManager:
    """Store and retrieve API keys from ~/.ember/config.yaml.

    Credentials are stored under providers.<provider>.api_key in the unified
    configuration file, eliminating the need for a separate credentials file.
    """

    config_dir: Path = field(default_factory=lambda: Path.home() / ".ember")
    config_file: Path = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config_dir, Path):
            self.config_dir = Path(self.config_dir)
        self.config_file = self.config_dir / "config.yaml"
        self._warn_legacy_credentials()

    def _warn_legacy_credentials(self) -> None:
        """Emit deprecation warning if legacy credentials file exists."""
        legacy_file = self.config_dir / "credentials"
        if legacy_file.exists():
            warnings.warn(
                f"Legacy credentials file {legacy_file} detected. "
                "Run 'ember configure migrate' to consolidate into config.yaml. "
                "The separate credentials file is deprecated and will be ignored "
                "in a future release.",
                DeprecationWarning,
                stacklevel=3,
            )

    def get_api_key(self, provider: str) -> str:
        """Return the stored API key for *provider* or raise if missing."""
        config = self._load_config()
        providers = config.get("providers", {})
        if not isinstance(providers, dict):
            raise CredentialNotFoundError(f"No credentials stored for provider '{provider}'")

        provider_cfg = providers.get(provider)
        if not isinstance(provider_cfg, dict):
            raise CredentialNotFoundError(f"No credentials stored for provider '{provider}'")

        api_key = provider_cfg.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            raise CredentialNotFoundError(f"Credential for '{provider}' is empty or invalid")

        # Skip unresolved env var placeholders
        cleaned = api_key.strip()
        if cleaned.startswith("${") and cleaned.endswith("}"):
            raise CredentialNotFoundError(
                f"Credential for '{provider}' contains unresolved placeholder: {cleaned}"
            )

        return cleaned

    def save_api_key(self, provider: str, api_key: str) -> None:
        """Persist *api_key* under providers.<provider>.api_key in config.yaml."""
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("Provider name must be a non-empty string")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("API key must be a non-empty string")

        cleaned_provider = provider.strip()
        cleaned_key = api_key.strip()

        if len(cleaned_key) < 5:
            raise ValueError("API key appears to be too short")
        if cleaned_key.startswith('"') and cleaned_key.endswith('"'):
            raise ValueError("API key should not be quoted")
        if " " in cleaned_key:
            raise ValueError("API key should not contain spaces")

        config = self._load_config()
        providers = config.setdefault("providers", {})
        if not isinstance(providers, dict):
            providers = {}
            config["providers"] = providers

        provider_cfg = providers.setdefault(cleaned_provider, {})
        if not isinstance(provider_cfg, dict):
            provider_cfg = {}
            providers[cleaned_provider] = provider_cfg

        provider_cfg["api_key"] = cleaned_key
        self._write_config(config)

    def delete(self, provider: str) -> None:
        """Remove stored credentials for *provider*."""
        config = self._load_config()
        providers = config.get("providers", {})
        if isinstance(providers, dict) and provider in providers:
            provider_cfg = providers[provider]
            if isinstance(provider_cfg, dict) and "api_key" in provider_cfg:
                del provider_cfg["api_key"]
                # Remove empty provider config
                if not provider_cfg:
                    del providers[provider]
                self._write_config(config)

    def list_providers(self) -> list[str]:
        """Return the providers that currently have stored credentials."""
        config = self._load_config()
        providers = config.get("providers", {})
        if not isinstance(providers, dict):
            return []

        result = []
        for name, cfg in providers.items():
            if isinstance(cfg, dict):
                api_key = cfg.get("api_key")
                if isinstance(api_key, str) and api_key.strip():
                    # Skip unresolved placeholders
                    if not (api_key.startswith("${") and api_key.endswith("}")):
                        result.append(name)
        return sorted(result)

    def get(self, provider: str) -> Optional[str]:
        """Legacy helper returning ``None`` instead of raising when missing."""
        try:
            return self.get_api_key(provider)
        except CredentialNotFoundError:
            return None

    def store(self, provider: str, api_key: str) -> None:
        """Backwards-compatible alias for :meth:`save_api_key`."""
        self.save_api_key(provider, api_key)

    # Internal helpers -------------------------------------------------
    def _load_config(self) -> Dict[str, Any]:
        """Load the config.yaml file, returning empty dict if missing."""
        if not self.config_file.exists():
            return {}

        try:
            with self.config_file.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise CredentialError(
                f"Config file {self.config_file} contains invalid YAML"
            ) from exc
        except OSError as exc:
            raise CredentialError(f"Unable to read {self.config_file}") from exc

        if data is None:
            return {}
        if not isinstance(data, dict):
            raise CredentialError(
                f"Config file {self.config_file} must contain a mapping"
            )
        return data

    def _write_config(self, config: Dict[str, Any]) -> None:
        """Atomically write config to config.yaml with secure permissions."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.config_dir), prefix="config-", suffix=".yaml.tmp"
        )
        os.close(fd)
        tmp_path = Path(tmp_name)

        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, self.config_file)
        finally:
            tmp_path.unlink(missing_ok=True)


_default_manager = CredentialManager()


def get_api_key(provider: str) -> str:
    """Convenience wrapper around the default CredentialManager."""
    return _default_manager.get_api_key(provider)


def save_api_key(provider: str, api_key: str) -> None:
    _default_manager.save_api_key(provider, api_key)


def delete_api_key(provider: str) -> None:
    _default_manager.delete(provider)


def list_providers() -> list[str]:
    return _default_manager.list_providers()


__all__ = [
    "CredentialError",
    "CredentialNotFoundError",
    "CredentialManager",
    "delete_api_key",
    "get_api_key",
    "list_providers",
    "save_api_key",
]
