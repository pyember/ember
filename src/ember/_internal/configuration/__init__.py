"""Configuration utilities for Ember's runtime.

This package exposes configuration management, schemas, and migration helpers
used internally to bootstrap the runtime context.
"""

from ember._internal.configuration.migrations import main as run_migrations
from ember._internal.configuration.migrations import migrate_config, migrate_credentials
from ember._internal.configuration.schemas import (
    CredentialEntry,
    CredentialsDict,
    EmberConfig,
    LoggingConfig,
    ModelsConfig,
    ProviderConfig,
)

__all__ = [
    "run_migrations",
    "migrate_config",
    "migrate_credentials",
    "CredentialsDict",
    "CredentialEntry",
    "EmberConfig",
    "LoggingConfig",
    "ModelsConfig",
    "ProviderConfig",
]
