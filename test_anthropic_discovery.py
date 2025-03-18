"""
Testing script for Anthropic model discovery and usage.
This verifies that our fix for the "'list' object has no attribute 'keys'" error is working.
"""

import logging
import time
import sys
from typing import Dict, Any

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Import discovery components
from ember.core.registry.model.providers.anthropic.anthropic_discovery import (
    AnthropicDiscovery,
)


def test_anthropic_discovery():
    """Test that we can discover Anthropic models without errors."""
    print("\n=== Testing Anthropic Discovery ===\n")

    try:
        # Create and initialize the provider
        discovery = AnthropicDiscovery()
        print("✅ Successfully created AnthropicDiscovery instance")

        # Fetch models
        print("Fetching Anthropic models...")
        models = discovery.fetch_models()

        print(f"✅ Successfully fetched {len(models)} Anthropic models")

        # Display some of the models
        print("\nSample models:")
        for model_id in list(models.keys())[:5]:
            print(f"  - {model_id}")

        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run tests with proper error reporting."""
    print("\n===== ANTHROPIC MODEL DISCOVERY TESTS =====\n")

    success = test_anthropic_discovery()

    print("\n===== TEST SUMMARY =====")
    print(f"Anthropic Discovery: {'✅ PASSED' if success else '❌ FAILED'}")

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
