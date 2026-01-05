"""Test that the setup wizard TypeScript code compiles successfully."""

import subprocess
import sys
import unittest
from pathlib import Path

import pytest


class TestSetupWizardTypeScript:
    """Test TypeScript compilation for setup wizard."""

    @pytest.fixture
    def setup_wizard_dir(self):
        """Get the setup wizard directory."""
        return Path(__file__).parent.parent.parent / "src" / "ember" / "cli" / "setup-wizard"

    def test_typescript_compiles(self, setup_wizard_dir):
        """Test that TypeScript code compiles without errors."""
        if not setup_wizard_dir.exists():
            pytest.skip("Setup wizard directory not found")

        # Check if npm is available
        try:
            subprocess.run(["npm", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("npm not available")

        # Run TypeScript compilation
        result = subprocess.run(
            ["npm", "run", "build"], cwd=setup_wizard_dir, capture_output=True, text=True
        )

        # Check for compilation errors
        assert result.returncode == 0, f"TypeScript compilation failed:\n{result.stderr}"

    def test_required_files_exist(self, setup_wizard_dir):
        """Test that all required TypeScript files exist."""
        required_files = [
            "tsconfig.json",
            "package.json",
            "src/cli.tsx",
            "src/components/SetupWizard.tsx",
            "src/components/LogoDisplay.tsx",
            "src/components/ProgressIndicator.tsx",
            "src/components/steps/Welcome.tsx",
            "src/components/steps/ProviderSelection.tsx",
            "src/components/steps/ApiKeySetup.tsx",
            "src/components/steps/TestConnection.tsx",
            "src/components/steps/Success.tsx",
            "src/types.ts",
            "src/utils/config.ts",
            "src/logos/index.ts",
        ]

        for file_path in required_files:
            full_path = setup_wizard_dir / file_path
            assert full_path.exists(), f"Required file missing: {file_path}"

    @unittest.skipIf(sys.platform.startswith("win"), "disabled on Windows")
    def test_no_typescript_errors_in_src(self, setup_wizard_dir):
        """Test that TypeScript type checking passes."""
        if not setup_wizard_dir.exists():
            pytest.skip("Setup wizard directory not found")

        # Run TypeScript type checking
        result = subprocess.run(
            ["npx", "tsc", "--noEmit"], cwd=setup_wizard_dir, capture_output=True, text=True
        )

        # Check for type errors
        assert result.returncode == 0, f"TypeScript type checking failed:\n{result.stderr}"

    @unittest.skipIf(sys.platform.startswith("win"), "disabled on Windows")
    def test_components_have_proper_types(self, setup_wizard_dir):
        """Test that component files have proper TypeScript types."""
        components_dir = setup_wizard_dir / "src" / "components"

        # Check that all component files have typed props
        for tsx_file in components_dir.rglob("*.tsx"):
            content = tsx_file.read_text()

            # Skip test files
            if "test" in tsx_file.name.lower():
                continue

            # Check for proper React.FC typing
            assert (
                "React.FC" in content or "FC<" in content
            ), f"{tsx_file.name} should have typed React components"

            # Check for interface definitions for props
            if "Props" in content:
                assert (
                    "interface" in content or "type" in content
                ), f"{tsx_file.name} should have typed props"
