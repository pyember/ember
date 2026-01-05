"""Utilities for migrating legacy Ember configuration and credentials.

Migrates from the old dual-file setup:
- ~/.ember/credentials (JSON) -> providers.<provider>.api_key in config.yaml
- ~/.ember/config.json -> config.yaml

After migration, all configuration lives in ~/.ember/config.yaml.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple

from ember._internal.context import EmberContext
from ember.core.utils.logging import get_logger

logger = get_logger(__name__)


def _create_backup(file_path: Path) -> Path:
    """Create a timestamped backup alongside ``file_path``.

    Args:
        file_path: Path to file that needs backing up.

    Returns:
        Path to the created backup file with timestamp suffix.

    Raises:
        OSError: If backup creation fails due to permissions or disk space.

        Note:
            Backup filename format: ``original.ext.bak.YYYYMMDD_HHMMSS``.
            For files without an extension the pattern is ``original.bak.YYYYMMDD_HHMMSS``.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # For files without extension, just add .bak.timestamp
    if file_path.suffix:
        backup = file_path.with_suffix(f"{file_path.suffix}.bak.{timestamp}")
    else:
        backup = file_path.with_suffix(f".bak.{timestamp}")
    shutil.copy2(file_path, backup)
    return backup


def migrate_credentials() -> bool:
    """Migrate credentials from ~/.ember/credentials to config.yaml.

    Reads the legacy JSON credentials file and writes each provider's API key
    to providers.<provider>.api_key in config.yaml.

    Returns:
        bool: ``True`` if migration was performed, otherwise ``False``.

    Note:
        The migration is idempotent and leaves a timestamped backup when it
        succeeds. A marker file prevents subsequent runs from re-importing the
        same data.
    """
    old_credentials = Path.home() / ".ember" / "credentials"

    if not old_credentials.exists():
        logger.debug("No legacy credentials file found")
        return False

    try:
        with open(old_credentials) as f:
            credentials = json.load(f)

        if not credentials:
            logger.debug("Legacy credentials file is empty")
            return False

        # Check if already migrated
        migration_marker = old_credentials.parent / ".credentials_migrated"
        if migration_marker.exists():
            logger.info("Credentials already migrated")
            return False

        # Migrate to config.yaml via EmberContext
        ctx = EmberContext.current()
        migrated_count = 0

        for provider, data in credentials.items():
            if isinstance(data, dict) and "api_key" in data:
                api_key = data["api_key"]
                if isinstance(api_key, str) and api_key.strip():
                    # Write directly to config.yaml path
                    ctx.set_config(f"providers.{provider}.api_key", api_key.strip())
                    migrated_count += 1
                    logger.info(f"Migrated credentials for {provider}")

        if migrated_count > 0:
            # Save all changes at once
            ctx.save()

            # Create backup and remove original
            backup = _create_backup(old_credentials)
            old_credentials.unlink()

            # Mark as migrated
            migration_marker.touch()

            print(f"Migrated {migrated_count} credential(s) to config.yaml. Backup: {backup}")
            return True

        return False

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in credentials file: {e}")
        return False
    except OSError as e:
        logger.error(f"File system error during migration: {e}")
        return False


def migrate_config() -> bool:
    """Migrate ``config.json`` to the YAML-based configuration format.

    Returns:
        bool: ``True`` when new values were migrated, ``False`` otherwise.
    """
    old_config = Path.home() / ".ember" / "config.json"

    if not old_config.exists():
        logger.debug("No legacy config file found")
        return False

    try:
        with open(old_config) as f:
            config = json.load(f)

        if not config:
            logger.debug("Legacy config file is empty")
            return False

        # Skip if new config already exists with content
        ctx = EmberContext.current()
        if ctx.get_config_path().exists() and ctx.get_all_config():
            logger.info("New config already exists, skipping migration")
            return False

        # Merge with existing config
        migrated_keys = []
        for key, value in config.items():
            if key not in ("version", "_migrated"):  # Skip metadata
                ctx.set_config(key, value)
                migrated_keys.append(key)
                logger.info(f"Migrated config: {key}")

        if migrated_keys:
            ctx.save()

            # Create backup and remove original
            backup = _create_backup(old_config)
            old_config.unlink()

            print(f"Migrated {len(migrated_keys)} config setting(s). Backup: {backup}")
            return True

        return False

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return False
    except OSError as e:
        logger.error(f"File system error during migration: {e}")
        return False


def main() -> Tuple[bool, bool]:
    """Run both credential and configuration migrations.

    Returns:
        Tuple[bool, bool]: Pair indicating whether each migration executed.
    """
    cred_migrated = migrate_credentials()
    config_migrated = migrate_config()

    if cred_migrated or config_migrated:
        logger.info("Migration completed successfully")
        print("Migration completed successfully.")
    else:
        logger.info("No migration needed")
        print("No migration needed.")

    return cred_migrated, config_migrated


if __name__ == "__main__":
    main()
