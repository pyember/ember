"""Unit tests for setup wizard functionality."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest


class TestSetupWizardConfig:
    """Test setup wizard configuration utilities."""

    def test_config_save_format(self, tmp_path):
        """Config is persisted under providers.<provider>.api_key."""
        from ember._internal.context.runtime import EmberContext

        EmberContext.reset()
        try:
            with patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=True):
                ctx = EmberContext.current()
                ctx.set_config("providers.openai.api_key", "sk-test123")
                ctx.save()

                config_file = tmp_path / ".ember" / "config.yaml"
                assert config_file.exists()
                content = config_file.read_text()
                assert "providers" in content
                assert "openai" in content
                assert "sk-test123" in content
        finally:
            EmberContext.reset()

    @unittest.skipIf(sys.platform.startswith("win"), "disabled on Windows")
    def test_credentials_save_secure(self, tmp_path):
        """Test that credentials are saved securely."""
        from ember._internal.context.runtime import EmberContext

        EmberContext.reset()
        try:
            with patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=True):
                ctx = EmberContext.current()
                ctx.set_config("providers.openai.api_key", "sk-test123")
                ctx.save()

                config_file = tmp_path / ".ember" / "config.yaml"
                assert config_file.exists()
                assert config_file.stat().st_mode & 0o777 == 0o600
        finally:
            EmberContext.reset()

    def test_provider_configuration_structure(self):
        """Test that provider configurations have correct structure."""
        # Since the setup wizard is TypeScript, we check the TypeScript source
        setup_wizard_dir = (
            Path(__file__).parent.parent.parent.parent / "src" / "ember" / "cli" / "setup-wizard"
        )
        types_file = setup_wizard_dir / "src" / "types.ts"

        if not types_file.exists():
            pytest.skip("Setup wizard types.ts not found")

        content = types_file.read_text()

        # Verify PROVIDERS constant exists
        assert "export const PROVIDERS" in content, "PROVIDERS constant should be exported"

        # Verify required fields in provider structure
        required_fields = ["name", "testModel", "description"]
        for field in required_fields:
            # Check for TypeScript property definition (field: type)
            assert f"{field}:" in content, f"Provider type should have {field} field"

        # Verify providers are defined
        for provider in ["openai", "anthropic", "google"]:
            assert (
                f'"{provider}"' in content or f"'{provider}'" in content
            ), f"Provider {provider} should be defined"

    def test_setup_mode_options(self):
        """Test that setup modes are properly defined."""
        setup_wizard_dir = (
            Path(__file__).parent.parent.parent.parent / "src" / "ember" / "cli" / "setup-wizard"
        )
        setup_mode_file = (
            setup_wizard_dir / "src" / "components" / "steps" / "SetupModeSelection.tsx"
        )

        if not setup_mode_file.exists():
            pytest.skip("SetupModeSelection.tsx not found")

        content = setup_mode_file.read_text()

        # Verify the modes are correctly defined
        assert (
            "'single'" in content or '"single"' in content
        ), "Setup mode 'single' should be defined"
        assert "'all'" in content or '"all"' in content, "Setup mode 'all' should be defined"

        # Verify type definition
        assert (
            "onSelectMode: (mode: 'single' | 'all')" in content
        ), "onSelectMode should accept 'single' | 'all' union type"
