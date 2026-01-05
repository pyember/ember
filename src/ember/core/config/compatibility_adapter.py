"""Configuration compatibility adapter for external AI tools.

This module provides transparent configuration compatibility between Ember
and external AI tools (OpenAI CLI, Anthropic CLI, etc.). It enables users
to use their existing configurations without modification while maintaining
full compatibility with Ember's configuration system.

Key features:
- Auto-detection of external configuration formats
- Runtime adaptation without modifying original files
- Preservation of tool-specific settings

The adapter follows the Adapter pattern to provide a unified interface
while supporting diverse configuration formats from the ecosystem.
"""

from collections.abc import Mapping


class CompatibilityAdapter:
    """Transparent adapter for external AI tool configuration formats.

    Provides detection, adaptation, and migration capabilities for configurations
    from external tools. Designed to work transparently at runtime without
    requiring user intervention.

    Attributes:
        EXTERNAL_FIELDS: Set of field names specific to external tools
            used for format detection.
    """

    # External tool-specific fields
    EXTERNAL_FIELDS: frozenset[str] = frozenset({"approvalMode", "fullAutoErrorMode", "notify"})

    @staticmethod
    def needs_adaptation(config: Mapping[str, object]) -> bool:
        """Detect if configuration needs adaptation from external format.

        Examines configuration structure and field names to determine if
        it originated from an external tool. Detection is based on:
        1. Presence of tool-specific fields (approvalMode, etc.)
        2. Provider configurations using envKey without api_key

        Args:
            config: Configuration dictionary to examine.

        Returns:
            bool: True if configuration appears to be from an external tool
                and needs adaptation, False if already in Ember format.

        Examples:
            >>> config = {"approvalMode": "auto", "providers": {...}}
            >>> CompatibilityAdapter.needs_adaptation(config)
            True

            >>> ember_config = {"providers": {"openai": {"api_key": "..."}}}
            >>> CompatibilityAdapter.needs_adaptation(ember_config)
            False
        """
        # Check for external tool-specific fields
        if CompatibilityAdapter.EXTERNAL_FIELDS & set(config):
            return True

        # Check for providers with envKey but no api_key
        providers = config.get("providers")
        if isinstance(providers, Mapping):
            for provider in providers.values():
                if (
                    isinstance(provider, Mapping)
                    and "envKey" in provider
                    and "api_key" not in provider
                ):
                    return True

        return False

    @staticmethod
    def adapt_provider_config(provider_config: Mapping[str, object]) -> dict[str, object]:
        """Adapt external provider format to Ember format at runtime.

        Transforms provider configuration from external tool format to Ember's
        expected format. Key transformations:
        1. Maps camelCase fields to snake_case (baseURL -> base_url)
        2. Normalizes apiKey -> api_key when present
        3. Preserves external envKey as env_key metadata

        Args:
            provider_config: Provider configuration in external format,
                typically containing envKey and baseURL fields.

        Returns:
            Provider configuration in Ember format. When the external config only
            specifies ``envKey`` (environment variable name), the resulting
            configuration will *not* include an ``api_key`` value. Ember does not
            hydrate provider credentials from environment variables.

        Runtime Behavior:
            - Environment variables are not used for credentials.
            - Legacy ``envKey`` fields are preserved as ``env_key`` metadata.
        """
        adapted: dict[str, object] = dict(provider_config)

        api_key = adapted.get("api_key")
        if api_key is None and "apiKey" in adapted:
            adapted["api_key"] = adapted["apiKey"]

        if "envKey" in adapted and "env_key" not in adapted:
            env_key = adapted.get("envKey")
            if isinstance(env_key, str) and env_key.strip():
                adapted["env_key"] = env_key.strip()

        # Map external fields to Ember fields
        field_mappings = {
            "baseURL": "base_url",  # Ember uses snake_case internally
        }

        for codex_field, ember_field in field_mappings.items():
            if codex_field in adapted and ember_field not in adapted:
                adapted[ember_field] = adapted[codex_field]

        return adapted

    @staticmethod
    def adapt_config(config: Mapping[str, object]) -> dict[str, object]:
        """Adapt complete external configuration to Ember format.

        Top-level adaptation method that processes entire configuration files.
        Handles both provider adaptations and preservation of tool-specific
        settings for potential future use.

        Args:
            config: Full configuration dictionary from external tool.

        Returns:
            Dict[str, Any]: Complete configuration adapted to Ember format:
                - All providers adapted to Ember schema
                - Tool-specific fields preserved in _external_compat
                - Original structure maintained where possible

        Processing:
            1. Checks if adaptation is needed
            2. Adapts each provider configuration
            3. Preserves external fields in _external_compat section
            4. Returns original config if no adaptation needed

        Note:
            This method is idempotent - can be called multiple times safely.
        """
        if not CompatibilityAdapter.needs_adaptation(config):
            return dict(config)

        adapted: dict[str, object] = dict(config)

        # Adapt all providers
        providers = adapted.get("providers")
        if isinstance(providers, Mapping):
            migrated_providers: dict[str, object] = {}
            for name, provider in providers.items():
                if isinstance(name, str) and isinstance(provider, Mapping):
                    migrated_providers[name] = CompatibilityAdapter.adapt_provider_config(provider)
                elif isinstance(name, str):
                    migrated_providers[name] = provider
            adapted["providers"] = migrated_providers

        # Preserve external tool-specific fields for potential extensions
        external_fields: dict[str, object] = {}
        for field in CompatibilityAdapter.EXTERNAL_FIELDS:
            if field in adapted:
                external_fields[field] = adapted[field]

        if external_fields:
            adapted["_external_compat"] = external_fields

        return adapted

    @staticmethod
    def migrate_provider(provider: Mapping[str, object]) -> dict[str, object]:
        """Migrate an external provider configuration to Ember format.

        Used for explicit migration commands (ember configure import) to
        permanently convert external configurations. Unlike adapt_provider_config,
        this creates a clean Ember configuration suitable for saving.

        Args:
            provider: Provider configuration in external format containing:
                - name: Provider display name
                - baseURL: API endpoint URL
                - envKey: Environment variable name for API key

        Returns:
            Clean Ember provider configuration. Credentials are imported only
            when the external configuration includes them inline (``apiKey`` or
            ``api_key``). Environment variable names (``envKey``) are preserved
            as metadata but never expanded into ``api_key`` placeholders.

        Migration Strategy:
            - Normalizes field naming for Ember consistency
            - Preserves original provider payload in ``_original``
            - Never hydrates credentials from environment variables

        Examples:
            >>> external = {"name": "OpenAI", "baseURL": "...", "envKey": "OPENAI_API_KEY"}
            >>> migrated = CompatibilityAdapter.migrate_provider(external)
            >>> "api_key" in migrated
            False
        """
        name = provider.get("name")
        base_url = provider.get("baseURL")

        api_key = provider.get("api_key")
        if api_key is None:
            api_key = provider.get("apiKey")

        ember_provider: dict[str, object] = {
            "name": name if isinstance(name, str) else "",
            "base_url": base_url if isinstance(base_url, str) else "",
            "_original": dict(provider),
        }

        env_key = provider.get("envKey")
        if isinstance(env_key, str) and env_key.strip():
            ember_provider["env_key"] = env_key.strip()

        if isinstance(api_key, str) and api_key.strip():
            ember_provider["api_key"] = api_key.strip()

        return ember_provider
