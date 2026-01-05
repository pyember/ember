"""Import external configuration into Ember's config format."""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

from ember._internal.context import EmberContext
from ember.core.config.compatibility_adapter import CompatibilityAdapter
from ember.core.config.loader import load_config, save_config


def cmd_import(args: argparse.Namespace) -> int:
    config_path = args.config_path
    output_path = args.output_path
    backup = args.backup
    dry_run = args.dry_run

    if not config_path:
        config_path = _find_external_config()
        if not config_path:
            print(
                "Error: no external configuration found. Please specify --config-path",
                file=sys.stderr,
            )
            return 1

    if not output_path:
        output_path = EmberContext.get_config_path()

    print(f"Importing config from: {config_path}")
    print(f"Target Ember config: {output_path}")

    external_config = load_config(config_path)

    if CompatibilityAdapter.needs_adaptation(external_config):
        print("Detected external configuration format. Adapting...")

    migrated_config = _migrate_config(external_config)

    if dry_run:
        print("\nMigrated configuration (dry run):")
        print(yaml.dump(migrated_config, default_flow_style=False))
        return 0

    if backup and output_path.exists():
        backup_path = _backup_config(output_path)
        print(f"Backed up existing config to: {backup_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_config(migrated_config, output_path)

    print(f"\nSuccessfully imported configuration to {output_path}")

    providers = migrated_config.get("providers")
    if isinstance(providers, dict):
        print("\nImported providers:")
        missing: list[str] = []
        for name, provider in providers.items():
            if not isinstance(name, str) or not isinstance(provider, dict):
                continue
            api_key = provider.get("api_key")
            if isinstance(api_key, str) and api_key.strip():
                status = "credential imported"
            else:
                status = "credential missing"
                missing.append(name)
            print(f"  - {name}: {status}")

        if missing:
            print(
                "\nCredentials are not imported from environment variables. "
                "Run `ember setup` or configure `providers.<provider>.api_key` via "
                "`ember configure`."
            )

    return 0


def _find_external_config() -> Path | None:
    """Search common locations for external AI tool configs."""
    config_dirs = [
        Path.home() / ".config" / "openai",
        Path.home() / ".openai",
        Path.home() / ".config" / "anthropic",
        Path.home() / ".anthropic",
    ]

    for config_dir in config_dirs:
        if not config_dir.exists():
            continue

        for filename in ["config.yaml", "config.yml", "config.json"]:
            config_path = config_dir / filename
            if config_path.exists():
                return config_path

    return None


def _migrate_config(external_config: dict[str, object]) -> dict[str, object]:
    """Adapt external tool configuration into Ember schema."""
    migrated: dict[str, object] = dict(external_config)

    providers = migrated.get("providers")
    if isinstance(providers, dict):
        migrated_providers: dict[str, object] = {}
        for name, provider in providers.items():
            if isinstance(name, str) and isinstance(provider, dict):
                migrated_providers[name] = CompatibilityAdapter.migrate_provider(provider)
            elif isinstance(name, str):
                migrated_providers[name] = provider
        migrated["providers"] = migrated_providers

    if "version" not in migrated:
        migrated["version"] = "1.0"

    migrated["_migration"] = {
        "from": "external",
        "date": datetime.now().isoformat(),
        "original_fields": {},
    }

    for field in CompatibilityAdapter.EXTERNAL_FIELDS:
        if field in external_config:
            migrated["_migration"]["original_fields"][field] = external_config[field]

    return migrated


def _backup_config(config_path: Path) -> Path:
    """Create a timestamped backup of an existing configuration file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = config_path.with_suffix(f".{timestamp}.backup{config_path.suffix}")
    shutil.copy2(config_path, backup_path)
    return backup_path
