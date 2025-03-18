#!/usr/bin/env python3
"""
Simple test script to verify that the model discovery service works correctly.
Includes thread safety and timeout handling improvements.
"""

import logging
import signal
import sys
import threading
import time
from contextlib import contextmanager

# Set up minimal logging to reduce output
logging.basicConfig(
    level=logging.INFO,  # Show more detailed logs for debugging
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

# Quiet noisy loggers
for logger_name in ["httpx", "httpcore", "openai"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Set ember's logger to WARNING to reduce verbosity but still show important messages
logging.getLogger("ember").setLevel(logging.WARNING)

from ember.core.registry.model.base.registry.discovery import ModelDiscoveryService


class TimeoutException(Exception):
    """Exception raised when a function execution times out."""

    pass


@contextmanager
def timeout(seconds, message="Execution timed out"):
    """A context manager that raises TimeoutException if execution takes too long."""

    def timeout_handler(signum, frame):
        raise TimeoutException(message)

    # Register the signal handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the original handler and cancel the alarm
        signal.signal(signal.SIGALRM, original_handler)
        signal.alarm(0)


def main():
    print("\n=== Testing Fixed Model Discovery Service ===\n")

    try:
        # Create the model discovery service
        start_time = time.time()
        print("Creating ModelDiscoveryService...")
        service = ModelDiscoveryService()
        print(f"Service created with {len(service.providers)} providers")

        # Discover models with global timeout protection
        print("\nDiscovering models (with timeout protection)...")

        try:
            with timeout(30, "Global timeout for discovery"):
                models = service.discover_models()

            duration = time.time() - start_time
            print(f"\nDiscovery complete in {duration:.2f}s")
            print(f"Found {len(models)} models")

            # Show sample of discovered models
            if models:
                print("\nSample of discovered models:")
                provider_counts = {}
                for model_id in models:
                    provider = model_id.split(":")[0] if ":" in model_id else "unknown"
                    provider_counts[provider] = provider_counts.get(provider, 0) + 1

                # Print summary by provider
                print("\nModels by provider:")
                for provider, count in provider_counts.items():
                    print(f"  {provider}: {count} models")

                # Show some examples
                print("\nSample models:")
                for i, model_id in enumerate(list(models.keys())[:5]):
                    print(f"  {i+1}. {model_id}")

            print("\n=== Test Complete ===")
            return len(models) > 0

        except TimeoutException as e:
            print(f"\n❌ ERROR: {e}")
            print(
                "The discovery process is taking too long. This indicates there may still be issues."
            )
            # Print active threads to help diagnose
            print("\nActive threads at timeout:")
            for t in threading.enumerate():
                print(f"  - {t.name}")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    print(f"\nTest result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)
