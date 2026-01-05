"""Diagnostic validator for test failure hypotheses."""

import importlib
import logging
import os
import sys
import threading
from typing import Any, Dict

# Configure diagnostic logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_diagnostics.log"), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


class DiagnosticValidator:
    """Validates hypotheses about test failures."""

    def __init__(self):
        self.results: Dict[str, Any] = {}

    def validate_ember_module_constraints(self):
        """Hypothesis 1: Check Ember Module field requirements."""
        logger.info("=== Validating Ember Module Constraints ===")

        try:
            from ember.core import EmberModule

            # Test 1: Can we create a basic module?
            class TestModule(EmberModule):
                pass

            instance = TestModule()
            logger.info("✓ Basic EmberModule creation successful")

            # Test 2: Can we set attributes without declaration?
            try:
                instance.dynamic_attr = "test"
                logger.error("✗ Dynamic attribute assignment allowed (unexpected)")
            except AttributeError as e:
                logger.info(f"✓ Dynamic attribute blocked as expected: {e}")

            # Test 3: Can we set with field declaration?
            class TestModuleWithField(EmberModule):
                declared_field: str = ""

            instance2 = TestModuleWithField()
            instance2.declared_field = "test"
            logger.info("✓ Declared field assignment successful")

            self.results["ember_constraints"] = True

        except Exception as e:
            logger.error(f"✗ Ember module validation failed: {e}")
            self.results["ember_constraints"] = False

    def validate_registry_structure(self):
        """Hypothesis 2: Check actual registry structure."""
        logger.info("=== Validating Registry Structure ===")

        try:
            import ember.api.data as data_api

            # Check what's actually available
            available_attrs = [attr for attr in dir(data_api) if not attr.startswith("_")]
            logger.info(f"Public data API attributes: {available_attrs}")

            # Check for _dataset_registry vs _registry
            has_dataset_registry = hasattr(data_api, "_dataset_registry")
            has_registry = hasattr(data_api, "_registry")

            logger.info(f"Has _dataset_registry: {has_dataset_registry}")
            logger.info(f"Has _registry: {has_registry}")

            if has_registry and not has_dataset_registry:
                logger.info("✓ Confirmed: Use _registry not _dataset_registry")
                self.results["registry_name"] = "_registry"
            else:
                self.results["registry_name"] = "_dataset_registry"

        except Exception as e:
            logger.error(f"✗ Registry validation failed: {e}")
            self.results["registry_structure"] = False

    def validate_mock_isolation(self):
        """Hypothesis 3: Check mock isolation effectiveness."""
        logger.info("=== Validating Mock Isolation ===")

        # Check if environment variables are leaking
        api_keys_before = {k: v for k, v in os.environ.items() if "API" in k or "KEY" in k}
        logger.info(f"API-related env vars: {list(api_keys_before.keys())}")

        # Check module import state
        imported_modules = [name for name in sys.modules if "ember" in name or "test" in name]
        logger.info(f"Ember/test modules loaded: {len(imported_modules)}")

        self.results["mock_isolation"] = {
            "env_vars": api_keys_before,
            "module_count": len(imported_modules),
        }

    def validate_thread_state(self):
        """Hypothesis 6: Check thread pool state."""
        logger.info("=== Validating Thread State ===")

        active_threads = threading.enumerate()
        thread_names = [t.name for t in active_threads]
        logger.info(f"Active threads ({len(active_threads)}): {thread_names}")

        # Check for lingering thread pools
        pool_threads = [t for t in thread_names if "pool" in t.lower()]
        if pool_threads:
            logger.warning(f"Found thread pool threads: {pool_threads}")

        self.results["thread_state"] = {"count": len(active_threads), "pool_threads": pool_threads}

    def validate_import_side_effects(self):
        """Hypothesis 7: Check import-time side effects."""
        logger.info("=== Validating Import Side Effects ===")

        # Track modules before import
        modules_before = set(sys.modules.keys())

        # Try importing suspect modules
        suspect_modules = [
            "ember.core.ember_context",
            "ember.models.model_registry",
            "ember.api.data",
        ]

        side_effects = {}
        for module_name in suspect_modules:
            try:
                if module_name in sys.modules:
                    del sys.modules[module_name]

                module = importlib.import_module(module_name)
                modules_after = set(sys.modules.keys())
                new_modules = modules_after - modules_before

                side_effects[module_name] = {
                    "new_modules": len(new_modules),
                    "has_init": hasattr(module, "__init__"),
                    "global_vars": len([k for k in dir(module) if not k.startswith("_")]),
                }

                logger.info(f"Module {module_name}: {side_effects[module_name]}")

            except Exception as e:
                logger.error(f"Failed to import {module_name}: {e}")
                side_effects[module_name] = {"error": str(e)}

        self.results["import_side_effects"] = side_effects

    def run_all_validations(self):
        """Run all hypothesis validations."""
        logger.info("Starting diagnostic validation...")

        self.validate_ember_module_constraints()
        self.validate_registry_structure()
        self.validate_mock_isolation()
        self.validate_thread_state()
        self.validate_import_side_effects()

        # Summary
        logger.info("\n=== VALIDATION SUMMARY ===")
        for key, value in self.results.items():
            logger.info(f"{key}: {value}")

        return self.results


def test_run_diagnostics():
    """Test runner for diagnostics."""
    validator = DiagnosticValidator()
    results = validator.run_all_validations()

    # Write results to file for analysis
    import json

    with open("diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\nDiagnostic results written to diagnostic_results.json")


if __name__ == "__main__":
    test_run_diagnostics()
