"""Runtime context management for Ember."""

from __future__ import annotations

import copy
import logging
import os
import tempfile
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ember.core.config.compatibility_adapter import CompatibilityAdapter
from ember.core.config.loader import load_config, save_config
from ember.core.credentials import CredentialManager, CredentialNotFoundError

if TYPE_CHECKING:
    from ember.core.utils.data.registry import DataRegistry
    from ember.models import ModelRegistry


logger = logging.getLogger(__name__)


class EmberContext:
    """Manage runtime configuration, credentials, and registries."""

    _thread_local = threading.local()
    _context_var: ContextVar["EmberContext | None"] = ContextVar("ember_context", default=None)
    _migration_lock = threading.Lock()
    _migration_checked = False

    def __init__(
        self,
        config_path: Path | None = None,
        parent: "EmberContext | None" = None,
        isolated: bool = False,
    ) -> None:
        """Initialize the context.

        Args:
            config_path: Explicit configuration file to load.
            parent: Parent context to clone for child contexts.
            isolated: Whether the context should avoid becoming the global default.
        """
        self._isolated = isolated
        self._lock = threading.RLock()
        self._credential_manager: "CredentialManager | None" = None
        self._model_registry: "ModelRegistry | None" = None
        self._data_registry: "DataRegistry | None" = None
        self._config: dict[str, object] = {}

        if parent is not None:
            self._config_file = parent._config_file
        else:
            candidate = Path(config_path) if config_path else self.get_config_path()
            if isolated and config_path is None and "EMBER_CONFIG_PATH" not in os.environ:
                temp_dir = Path(tempfile.gettempdir()) / f"ember-isolated-{uuid4().hex}"
                candidate = temp_dir / "config.yaml"
            self._config_file = candidate

        if config_path:
            self._config = self._load_config_from_path(self._config_file)
        elif parent is not None:
            self._config = copy.deepcopy(parent._config)
        elif isolated:
            self._config = self._load_config_if_exists(self._config_file)
        else:
            self._config = self._load_default_config()

        if not isolated and parent is None:
            cls = self.__class__
            cls._thread_local.context = self
            cls._context_var.set(self)
            self._check_migration()

    @classmethod
    def current(cls) -> "EmberContext":
        """Return the active context, creating one if needed.

        Returns:
            EmberContext: Active context for the current thread or async task.
        """
        ctx = cls._context_var.get()
        if ctx is not None:
            return ctx

        stored = vars(cls._thread_local).get("context")
        if stored is not None:
            cls._context_var.set(stored)
            return stored

        ctx = cls()
        cls._thread_local.context = ctx
        cls._context_var.set(ctx)
        return ctx

    @property
    def credential_manager(self) -> CredentialManager:
        """Return the credential manager for this context.

        Returns:
            CredentialManager: Credential manager bound to this context.
        """
        if self._credential_manager is None:
            with self._lock:
                if self._credential_manager is None:
                    self._credential_manager = CredentialManager(self._config_file.parent)
        return self._credential_manager

    @property
    def _credentials(self) -> CredentialManager:
        """Alias for legacy credential manager access.

        Returns:
            CredentialManager: Credential manager bound to this context.
        """
        return self.credential_manager

    @property
    def model_registry(self) -> "ModelRegistry":
        """Return the model registry bound to this context.

        Returns:
            ModelRegistry: Registry configured against this context.
        """
        if self._model_registry is None:
            with self._lock:
                if self._model_registry is None:
                    from ember.models import ModelRegistry

                    self._model_registry = ModelRegistry(context=self)
        return self._model_registry

    @property
    def data_registry(self) -> "DataRegistry":
        """Return the data registry bound to this context.

        Returns:
            DataRegistry: Registry configured against this context.
        """
        if self._data_registry is None:
            with self._lock:
                if self._data_registry is None:
                    from ember.core.utils.data.registry import DataRegistry

                    self._data_registry = DataRegistry(context=self)
        return self._data_registry

    def get_credential(self, provider: str, env_var: str | None = None) -> str:
        """Return the configured credential for ``provider``.

        Credentials are resolved exclusively from the centralized Ember
        configuration.

        The ``env_var`` parameter is accepted for backward compatibility and is
        ignored.
        """

        providers = self._config.get("providers")
        if providers is not None and not isinstance(providers, dict):
            raise TypeError(
                f"Expected providers to be a mapping, got {type(providers).__name__}"
            )
        if isinstance(providers, dict):
            if provider in providers:
                provider_cfg = providers.get(provider)
                if provider_cfg is not None and not isinstance(provider_cfg, dict):
                    raise TypeError(
                        f"Expected providers.{provider} to be a mapping, got "
                        f"{type(provider_cfg).__name__}"
                    )
                if not isinstance(provider_cfg, dict):
                    config_path = self._config_file or self.get_config_path()
                    raise CredentialNotFoundError(
                        "\n".join(
                            [
                                f"Missing credential for provider '{provider}'.",
                                (
                                    f"Set providers.{provider}.api_key in {config_path} "
                                    "(use `ember configure`)."
                                ),
                            ]
                        )
                    )
                api_key = provider_cfg.get("api_key")
                if isinstance(api_key, str):
                    cleaned = api_key.strip()
                    if cleaned.startswith("${") and cleaned.endswith("}"):
                        raise CredentialNotFoundError(
                            "\n".join(
                                [
                                    f"Missing credential for provider '{provider}'.",
                                    (
                                        f"Found unresolved placeholder providers.{provider}."
                                        f"api_key={cleaned}."
                                    ),
                                    (
                                        "Ember does not expand environment variables inside config "
                                        "files."
                                    ),
                                    (
                                        f"Set providers.{provider}.api_key to a real credential "
                                        "(use `ember setup`)."
                                    ),
                                ]
                            )
                        )
                    if cleaned:
                        return cleaned

                config_path = self._config_file or self.get_config_path()
                raise CredentialNotFoundError(
                    "\n".join(
                        [
                            f"Missing credential for provider '{provider}'.",
                            (
                                f"Set providers.{provider}.api_key in {config_path} "
                                "(use `ember configure`)."
                            ),
                        ]
                    )
                )

        try:
            return self.credential_manager.get_api_key(provider)
        except CredentialNotFoundError as exc:
            config_path = self._config_file or self.get_config_path()
            raise CredentialNotFoundError(
                "\n".join(
                    [
                        f"Missing credential for provider '{provider}'.",
                        f"Set providers.{provider}.api_key in {config_path} "
                        "(use `ember configure`).",
                    ]
                )
            ) from exc

    def get_config(self, key: str, default: Any = None) -> Any:
        """Return configuration for ``key`` using dotted traversal.

        Args:
            key: Dotted configuration path (for example ``"providers.openai"``).
            default: Fallback value returned when the key is missing.

        Returns:
            Any: Configured value or ``default`` if the key is absent.
        """
        if not isinstance(key, str) or not key.strip():
            return default

        value: Any = self._config
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def set_config(self, key: str, value: Any) -> None:
        """Persist configuration ``value`` at dotted path ``key``.

        Args:
            key: Dotted configuration path to update.
            value: Value to assign.

        Raises:
            TypeError: If ``key`` is not a string.
            ValueError: If ``key`` is empty or malformed.
        """
        if not isinstance(key, str):
            raise TypeError(f"Configuration key must be a string, got {type(key).__name__}")
        if not key or not key.strip():
            raise ValueError("Configuration key cannot be empty")
        if any(not part for part in key.split(".")):
            raise ValueError(f"Invalid configuration key: '{key}'")

        parts = key.split(".")
        with self._lock:
            cursor = self._config
            for part in parts[:-1]:
                next_value = cursor.get(part)
                if not isinstance(next_value, dict):
                    next_value = {}
                cursor[part] = next_value
                cursor = next_value
            cursor[parts[-1]] = value

    def get_model(self, model_id: str | None = None, **kwargs: Any) -> Any:
        """Return a configured model instance.

        Args:
            model_id: Explicit model identifier to load. Defaults to ``models.default``.
            **kwargs: Provider-specific keyword arguments forwarded to the registry.

        Returns:
            Any: Model instance supplied by the registry.
        """
        selected = model_id or self.get_config("models.default", "gpt-3.5-turbo")
        return self.model_registry.get_model(selected, **kwargs)

    def list_models(self) -> list[str]:
        """Return all models discoverable via the catalog.

        Returns:
            list[str]: Available model identifiers.
        """
        from ember.models.catalog import list_available_models

        return list_available_models()

    @classmethod
    def reset(cls) -> None:
        """Clear process-local context state."""
        vars(cls._thread_local).pop("context", None)
        cls._context_var.set(None)

    def get_all_config(self) -> dict[str, object]:
        """Return a shallow copy of the current configuration.

        Returns:
            dict[str, Any]: Copy of the configuration state.
        """
        return dict(self._config)

    def reload(self) -> None:
        """Reload configuration from disk.

        Missing files fall back to the default config path.
        """
        with self._lock:
            config_file = self._config_file
            if config_file and config_file.exists():
                self._config = self._load_config_from_path(config_file)
            else:
                self._config = self._load_default_config()

    def save(self) -> None:
        """Persist configuration to disk.

        Raises:
            RuntimeError: If the context has no associated configuration file.
        """
        if not self._config_file:
            raise RuntimeError("Configuration path is undefined for this context")
        with self._lock:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            save_config(self._config, self._config_file)

    def load_dataset(self, name: str, **kwargs: Any) -> Any:
        """Load a dataset via the data registry.

        Args:
            name: Dataset identifier registered with the data registry.
            **kwargs: Loader-specific keyword arguments.

        Returns:
            Any: Dataset instance provided by the registry.
        """
        return self.data_registry.load(name, **kwargs)

    def create_child(self, **config_overrides: Any) -> "EmberContext":
        """Create an isolated child context with overrides applied.

        Args:
            **config_overrides: Key-value pairs to apply to the child configuration.

        Returns:
            EmberContext: Newly created child context.
        """
        child = EmberContext(parent=self, isolated=True)
        for key, override in config_overrides.items():
            if isinstance(override, dict):
                for nested_key, nested_value in override.items():
                    child.set_config(f"{key}.{nested_key}", nested_value)
            else:
                child.set_config(key, override)
        return child

    def __enter__(self) -> "EmberContext":
        """Activate this context for ``with`` statements.

        Returns:
            EmberContext: The activated context.
        """
        cls = self.__class__
        thread_state = vars(cls._thread_local)
        self._had_thread_context = "context" in thread_state
        self._previous_thread = thread_state.get("context")
        self._token = cls._context_var.set(self)
        cls._thread_local.context = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: D401
        """Restore the previous context after ``with`` blocks.

        Args:
            exc_type: Exception type raised inside the context block.
            exc_val: Exception instance raised inside the context block.
            exc_tb: Traceback linked to the exception raised inside the block.
        """
        cls = self.__class__
        if self._had_thread_context:
            cls._thread_local.context = self._previous_thread
        else:
            vars(cls._thread_local).pop("context", None)
        cls._context_var.reset(self._token)

    @staticmethod
    def get_config_path() -> Path:
        """Return the resolved config file path.

        Returns:
            Path: Absolute path to the configuration file.
        """
        env_path = os.environ.get("EMBER_CONFIG_PATH")
        if not env_path:
            return Path.home() / ".ember" / "config.yaml"
        path = Path(env_path)
        if path.is_dir() or not path.suffix:
            return path / "config.yaml"
        return path

    def _load_default_config(self) -> dict[str, object]:
        """Load configuration from the default path when available.

        Returns:
            dict[str, Any]: Configuration data or an empty dictionary if absent.
        """
        config_file = self.get_config_path()
        if not config_file.exists():
            return {}
        return self._load_config_from_path(config_file)

    def _load_config_if_exists(self, path: Path | None) -> dict[str, object]:
        """Load configuration only when the provided path exists.

        Args:
            path: Path to a configuration file.

        Returns:
            dict[str, Any]: Loaded configuration or an empty dictionary.
        """
        if path and path.exists():
            return self._load_config_from_path(path)
        return {}

    def _load_config_from_path(self, path: Path) -> dict[str, object]:
        """Load and normalize configuration from ``path``.

        Args:
            path: Path pointing to a configuration file.

        Returns:
            dict[str, Any]: Normalized configuration dictionary.
        """
        config = load_config(str(path))
        return CompatibilityAdapter.adapt_config(config)

    def _check_migration(self) -> None:
        """Run legacy configuration migrations once per process."""
        cls = self.__class__
        with cls._migration_lock:
            if cls._migration_checked:
                return
            cls._migration_checked = True

        old_creds = Path.home() / ".ember" / "credentials"
        old_config = Path.home() / ".ember" / "config.json"
        if not old_creds.exists() and not old_config.exists():
            return

        from ember._internal.configuration.migrations import migrate_config, migrate_credentials

        migrate_credentials()
        migrate_config()


def current_context() -> EmberContext:
    """Return the active Ember context.

    Returns:
        EmberContext: Active context for the caller.
    """
    return EmberContext.current()
